"""
SSL Baseline 评估脚本 — 在 Qwen2.5-VL 上复现 SSL (EMNLP 2025 Findings)。

SSL 核心思想 (Hua et al., EMNLP 2025):
  用 SAE 找到 faithful direction (d_faithful) 和 hallucinated direction (d_hall)，
  推理时做双阶段 steering：
    - Prefill: image tokens += alpha * d_faithful  (增强忠实方向)
    - Decoding: generated tokens -= alpha * d_hall   (抑制幻觉方向)
  其中 alpha = gamma * ||x|| / ||d||  (ASP: Adaptive Steering Parameters)

与 SAVE 的区别：
  - SAVE: 只在一个方向做 steering，强度固定
  - SSL: 双方向 (prefill + decoding)，强度自适应 (ASP)

与 NeuronEye 的区别：
  - SSL: 固定的 steering 方向，不看 query，全局作用
  - NeuronEye: query-conditioned routing，局部作用于特定 patches

用法：
    # 先识别 faithful/hallucinated features，再评估
    python eval/eval_ssl.py --mode both --benchmarks cvbench mmstar
    python eval_new/eval_ssl.py --mode both --gamma 0.2 --steer_layer 8

    # 分步
    python eval/eval_ssl.py --mode identify --identify_samples 500
    python eval/eval_ssl.py --mode eval --gamma 0.2 --benchmarks cvbench blink
"""
import os, sys, re, json, random, argparse
from collections import defaultdict
from functools import wraps
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from src.SAE import SAE

try:
    from config import CFG
except ImportError:
    class CFG:
        model_id="Qwen/Qwen2.5-VL-7B-Instruct"; save_dir="output/qwen_old/sae_ckpt"
        vis_layer=8; latent_mult=32; topk=32; device="cuda"

ALL_BLINK_SUBTASKS=["Art_Style","Counting","Forensic_Detection","Functional_Correspondence","IQ_Test","Jigsaw","Multi-view_Reasoning","Object_Localization","Relative_Depth","Relative_Reflectance","Semantic_Correspondence","Spatial_Relation","Visual_Correspondence","Visual_Similarity"]
DEPTH_SPATIAL_SUBTASKS=["Relative_Depth","Spatial_Relation","Multi-view_Reasoning","Object_Localization"]


class SSLSteering:
    """
    SSL: 双阶段 SAE-based steering with Adaptive Steering Parameters。

    Feature Identification (Cohen's d):
      对每个 SAE feature j，比较正确/错误回答时的激活分布：
        d_j = (mean_correct - mean_incorrect) / pooled_std
      d_j > 0 且最大 → faithful feature
      d_j < 0 且最小 → hallucinated feature

    Steering (ASP):
      Prefill: vision tokens += (gamma * ||x|| / ||d_f||) * d_faithful
      Decoding: gen tokens -= (gamma * ||x|| / ||d_h||) * d_hall
    """

    def __init__(self, model, sae, processor=None, steer_layer=8, gamma=0.2):
        self.model = model
        self.sae = sae
        self.processor = processor
        self.steer_layer = steer_layer
        self.gamma = gamma

        self.d_faithful = None   # (1, 1, dim) faithful steering direction
        self.d_hall = None       # (1, 1, dim) hallucinated steering direction
        self.faithful_idx = None
        self.hall_idx = None

        self._hook_handle = None
        self._vision_pos = None
        self._num_img_tokens = None

    def identify_features(self, correct_acts, incorrect_acts):
        """用 Cohen's d 找 faithful / hallucinated features。"""
        latent_dim = correct_acts[0].shape[-1]

        # 计算每个 feature 在 correct/incorrect 中的平均激活
        correct_means = torch.zeros(latent_dim)
        correct_vars = torch.zeros(latent_dim)
        incorrect_means = torch.zeros(latent_dim)
        incorrect_vars = torch.zeros(latent_dim)

        for act in correct_acts:
            m = act.mean(dim=0).cpu()
            correct_means += m
        correct_means /= max(len(correct_acts), 1)

        for act in incorrect_acts:
            m = act.mean(dim=0).cpu()
            incorrect_means += m
        incorrect_means /= max(len(incorrect_acts), 1)

        # 计算方差
        for act in correct_acts:
            m = act.mean(dim=0).cpu()
            correct_vars += (m - correct_means) ** 2
        correct_vars /= max(len(correct_acts) - 1, 1)

        for act in incorrect_acts:
            m = act.mean(dim=0).cpu()
            incorrect_vars += (m - incorrect_means) ** 2
        incorrect_vars /= max(len(incorrect_acts) - 1, 1)

        # Pooled std
        n1, n2 = len(correct_acts), len(incorrect_acts)
        pooled_var = ((n1 - 1) * correct_vars + (n2 - 1) * incorrect_vars) / max(n1 + n2 - 2, 1)
        pooled_std = (pooled_var + 1e-8).sqrt()

        # Cohen's d
        cohens_d = (correct_means - incorrect_means) / pooled_std

        # Faithful: largest positive Cohen's d
        self.faithful_idx = cohens_d.argmax().item()
        # Hallucinated: largest negative Cohen's d (smallest value)
        self.hall_idx = cohens_d.argmin().item()

        print(f"  Faithful feature: index={self.faithful_idx}, Cohen's d={cohens_d[self.faithful_idx]:.4f}")
        print(f"  Hallucinated feature: index={self.hall_idx}, Cohen's d={cohens_d[self.hall_idx]:.4f}")

        # 提取 SAE decoder 列作为 steering 方向
        device = next(self.sae.parameters()).device
        self.d_faithful = self.sae.decoder.weight[:, self.faithful_idx].detach().clone().view(1, 1, -1)
        self.d_hall = self.sae.decoder.weight[:, self.hall_idx].detach().clone().view(1, 1, -1)

        return {
            "faithful_idx": self.faithful_idx,
            "hall_idx": self.hall_idx,
            "faithful_cohens_d": float(cohens_d[self.faithful_idx]),
            "hall_cohens_d": float(cohens_d[self.hall_idx]),
        }

    def save_features(self, path):
        torch.save({
            "faithful_idx": self.faithful_idx, "hall_idx": self.hall_idx,
            "d_faithful": self.d_faithful, "d_hall": self.d_hall,
        }, path)
        print(f"  Features saved: {path}")

    def load_features(self, path):
        data = torch.load(path, map_location="cpu")
        self.faithful_idx = data["faithful_idx"]
        self.hall_idx = data["hall_idx"]
        self.d_faithful = data["d_faithful"]
        self.d_hall = data["d_hall"]
        print(f"  Loaded: faithful={self.faithful_idx}, hall={self.hall_idx}")

    def _set_positions(self, input_ids, image_grid_thw):
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        spatial_merge = self.model.config.vision_config.spatial_merge_size
        ids = input_ids[0]
        vs = (ids == vision_start_id).nonzero(as_tuple=True)[0]
        if len(vs) > 0:
            self._vision_pos = vs[0].item() + 1
            self._num_img_tokens = int(
                image_grid_thw[0, 1] * image_grid_thw[0, 2] / (spatial_merge ** 2)
            )
        else:
            self._vision_pos = None
            self._num_img_tokens = None

    def install_hook(self):
        layers = self.model.model.language_model.layers
        self._hook_handle = layers[self.steer_layer].register_forward_hook(self._ssl_hook)

    def remove_hook(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    def _ssl_hook(self, module, args, outputs):
        """SSL 双阶段 steering hook (原版逻辑)。"""
        if self.d_faithful is None or self.d_hall is None:
            return outputs

        with torch.no_grad():
            if isinstance(outputs, tuple):
                unpack = list(outputs)
            else:
                unpack = [outputs]

            device = unpack[0].device
            d_f = self.d_faithful.to(device).to(unpack[0].dtype)
            d_h = self.d_hall.to(device).to(unpack[0].dtype)

            if unpack[0].shape[1] != 1:
                # Prefill: 对 image tokens 加 faithful direction
                if self._vision_pos is not None and self._num_img_tokens is not None:
                    vp = self._vision_pos
                    ni = self._num_img_tokens
                    if vp + ni <= unpack[0].shape[1]:
                        x_img = unpack[0][:, vp:vp + ni, :]
                        x_norm = x_img.norm(dim=-1, keepdim=True)
                        d_norm = d_f.norm() + 1e-6
                        alpha = self.gamma * x_norm / d_norm
                        unpack[0][:, vp:vp + ni, :] = x_img + alpha * d_f
            else:
                # Decoding: 对生成 token 减 hallucinated direction
                x_gen = unpack[0]
                x_norm = x_gen.norm()
                d_norm = d_h.norm() + 1e-6
                alpha = self.gamma * x_norm / d_norm
                unpack[0] = x_gen - alpha * d_h

            return tuple(unpack) if isinstance(outputs, tuple) else unpack[0]

    def patch_generate(self):
        original_generate = self.model.generate

        @wraps(original_generate)
        def patched_generate(*args, **kwargs):
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            image_grid_thw = kwargs.get("image_grid_thw", None)
            if input_ids is not None and image_grid_thw is not None:
                self._set_positions(input_ids, image_grid_thw)

            self.install_hook()
            try:
                result = original_generate(*args, **kwargs)
            finally:
                self.remove_hook()
            return result

        self.model.generate = patched_generate
        self._original_generate = original_generate


# ══════════════════════════════════════════════════════════════════════════════
# Feature Identification
# ══════════════════════════════════════════════════════════════════════════════

def identify_ssl_features(model, processor, sae, steer_layer, max_samples=500, save_path=None):
    from qwen_vl_utils import process_vision_info

    print(f"\n{'='*60}\n  SSL: Identifying Features (Cohen's d)\n{'='*60}")

    ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    if max_samples: ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Calibration samples: {len(ds)}")

    layers = model.model.language_model.layers
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    spatial_merge = model.config.vision_config.spatial_merge_size
    sae_device = next(sae.parameters()).device

    correct_acts, incorrect_acts = [], []

    for item in tqdm(ds, desc="Collecting activations"):
        image = item["image"]
        choices = item["choices"]
        answer = item["answer"]
        ct = "\n".join([f"{chr(ord('A')+i)}. {c}" for i,c in enumerate(choices)])
        prompt = f"Answer the following question.\nSelect the correct option and output ONLY the letter.\nDo NOT output explanation.\n\n{item['prompt']}\n\nChoices:\n{ct}\n\nAnswer:"

        tmp_path = "/tmp/ssl_cal_tmp.png"
        if isinstance(image, Image.Image): image.save(tmp_path)
        else: tmp_path = image

        try:
            messages = [{"role":"user","content":[{"type":"image","image":f"file://{tmp_path}"},{"type":"text","text":prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

            input_ids = inputs["input_ids"][0]
            image_grid = inputs["image_grid_thw"]
            num_img = int(image_grid[0,1]*image_grid[0,2]/(spatial_merge**2))
            vs_pos = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
            if len(vs_pos) == 0: continue
            vision_pos = vs_pos[0].item() + 1

            collected = {}; done = [False]
            def make_hook(li):
                def hf(m, a, o):
                    if done[0]: return
                    hs = o[0] if isinstance(o, tuple) else o
                    if hs.shape[1] > 1: collected[li] = hs.detach(); done[0] = True
                return hf

            handle = layers[steer_layer].register_forward_hook(make_hook(steer_layer))
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            handle.remove()

            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(out_ids[0, input_len:], skip_special_tokens=True).strip()

            text_upper = response.strip().split("\n")[0].strip().upper()
            extracted = None
            m = re.search(r'\(([A-Z])\)', text_upper)
            if m: extracted = f"({m.group(1)})"
            else:
                m = re.match(r'^([A-Z])(?:[\s.,):]|$)', text_upper)
                if m: extracted = f"({m.group(1)})"
            is_correct = (extracted == answer)

            if steer_layer in collected:
                hs = collected[steer_layer]
                if vision_pos + num_img <= hs.shape[1]:
                    vision_hs = hs[0, vision_pos:vision_pos+num_img, :].float()
                    with torch.no_grad():
                        z = sae.encode(vision_hs.to(sae_device))
                    if is_correct: correct_acts.append(z.cpu())
                    else: incorrect_acts.append(z.cpu())

        except Exception as e:
            continue

    print(f"  Correct: {len(correct_acts)}, Incorrect: {len(incorrect_acts)}")
    if not correct_acts or not incorrect_acts:
        print("  ERROR: Not enough samples!"); return None

    ssl = SSLSteering(model, sae, processor=processor, steer_layer=steer_layer)
    info = ssl.identify_features(correct_acts, incorrect_acts)
    if save_path: ssl.save_features(save_path)
    return ssl


# ══════════════════════════════════════════════════════════════════════════════
# Model loading & generate
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_sae(model_id, sae_dir, layer, latent_mult=32, topk=32):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print(f"Loading Qwen2.5-VL: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, max_pixels=1280*28*28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
    print(f"  Layers: {len(model.model.language_model.layers)}")

    dim = model.config.text_config.hidden_size
    device = next(model.parameters()).device
    sae = SAE(dim, dim*latent_mult, topk).float().to(device)
    sae_path = os.path.join(sae_dir, f"sae_layer{layer}.pt")
    sae.load_state_dict(torch.load(sae_path, map_location=device)); sae.eval()
    print(f"  SAE: {sae_path}")
    return model, processor, sae

def extract_choice_letter(response, max_letter="Z"):
    text=response.strip().split("\n")[0].strip().upper()
    if not text: return None
    pat=f'[A-{max_letter}]'
    m=re.search(rf'\(({pat})\)',text)
    if m: return m.group(1)
    m=re.search(rf'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?({pat})\)?',text)
    if m: return m.group(1)
    m=re.match(rf'^({pat})(?:[\s.,):]|$)',text)
    if m: return m.group(1)
    return None

def qwen_generate(model, processor, image, question, max_new_tokens=32):
    try:
        if isinstance(image, str): image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image): image = image.convert("RGB")
        else: return ""
        messages=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":question}]}]
        inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt").to(model.device)
        with torch.no_grad():
            out=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
        il=inputs["input_ids"].shape[1]
        return processor.decode(out[0,il:],skip_special_tokens=True).strip()
    except Exception as e:
        print(f"  [error]: {e}"); torch.cuda.empty_cache(); return ""


# ══════════════════════════════════════════════════════════════════════════════
# Benchmarks
# ══════════════════════════════════════════════════════════════════════════════

def eval_cvbench(model, processor, max_samples=None):
    print(f"\n{'='*60}\n  CV-Bench (SSL)\n{'='*60}")
    ds=load_dataset("nyu-visionx/CV-Bench",split="test")
    if max_samples: ds=ds.select(range(min(max_samples,len(ds))))
    results=[]
    for item in tqdm(ds,desc="CV-Bench"):
        ch=item["choices"]; ct="\n".join([f"{chr(ord('A')+i)}. {c}" for i,c in enumerate(ch)])
        prompt=f"Answer the following question.\nSelect the correct option and output ONLY the letter.\nDo NOT output explanation.\n\n{item['prompt']}\n\nChoices:\n{ct}\n\nAnswer:"
        resp=qwen_generate(model,processor,item["image"],prompt)
        ext=extract_choice_letter(resp); ans=item["answer"]
        ef=f"({ext})" if ext else None
        results.append({"idx":item["idx"],"task":item["task"],"source":item["source"],"type":item["type"],"answer":ans,"response":resp,"extracted":ef,"correct":ef==ans})

    bs=defaultdict(list); bt=defaultdict(list)
    for r in results: bs[r["source"]].append(r); bt[r["task"]].append(r)
    def acc(it): return sum(1 for i in it if i["correct"])/len(it)*100 if it else 0
    aa=acc(bs.get("ADE20K",[])); ac=acc(bs.get("COCO",[])); ao=acc(bs.get("Omni3D",[]))
    a2=(aa+ac)/2; a3=ao; cv=(a2+a3)/2; nu=sum(1 for r in results if r["extracted"] is None)
    print(f"\n  Overall: {cv:.2f}  2D: {a2:.2f}  3D: {a3:.2f}  Unparsed: {nu}/{len(results)}")
    for t in sorted(bt): print(f"    {t:20s}: {acc(bt[t]):6.2f} (n={len(bt[t])})")
    return {"benchmark":"CV-Bench","cv_bench":cv,"acc_2d":a2,"acc_3d":a3,"per_task":{t:acc(it) for t,it in bt.items()},"n_unparsed":nu,"n_total":len(results)}, results

def eval_mmstar(model, processor, max_samples=None):
    print(f"\n{'='*60}\n  MMStar (SSL)\n{'='*60}")
    ds=load_dataset("Lin-Chen/MMStar",split="val")
    if max_samples: ds=ds.select(range(min(max_samples,len(ds))))
    results=[]
    for item in tqdm(ds,desc="MMStar"):
        prompt=f"Answer the following question.\nSelect the correct option and output ONLY the letter (A, B, C, or D).\nDo NOT output explanation.\n\n{item['question']}\n\nAnswer:"
        resp=qwen_generate(model,processor,item["image"],prompt)
        ext=extract_choice_letter(resp,max_letter="D"); ans=item["answer"].strip().upper()
        results.append({"index":item["index"],"category":item["category"],"l2_category":item["l2_category"],"answer":ans,"response":resp,"extracted":ext,"correct":ext==ans})
    def acc(it): return sum(1 for i in it if i["correct"])/len(it)*100 if it else 0
    ov=acc(results); nu=sum(1 for r in results if r["extracted"] is None)
    print(f"\n  Overall: {ov:.2f}  Unparsed: {nu}/{len(results)}")
    return {"benchmark":"MMStar","overall":ov,"n_unparsed":nu,"n_total":len(results)}, results

def eval_realworldqa(model, processor, max_samples=None):
    print(f"\n{'='*60}\n  RealWorldQA (SSL)\n{'='*60}")
    ds=load_dataset("xai-org/RealworldQA",split="test")
    if max_samples: ds=ds.select(range(min(max_samples,len(ds))))
    results=[]
    for i,item in enumerate(tqdm(ds,desc="RealWorldQA")):
        q=item["question"]; a=item["answer"]; resp=qwen_generate(model,processor,item["image"],q)
        ans=a.strip()
        if ans in ("A","B","C","D"):
            ext=extract_choice_letter(resp,max_letter="D"); correct=ext==ans
        elif ans.lower() in ("yes","no"):
            rl=resp.strip().lower(); correct=(rl.startswith("yes") and ans.lower()=="yes") or (rl.startswith("no") and ans.lower()=="no"); ext=resp[:50]
        else:
            rn=resp.strip().split("\n")[0].strip().lower().rstrip("."); an=ans.lower().rstrip(".")
            correct=rn==an or rn.startswith(an); ext=resp[:50]
        results.append({"idx":i,"answer":a,"response":resp,"extracted":ext,"correct":correct})
    def acc(it): return sum(1 for i in it if i["correct"])/len(it)*100 if it else 0
    ov=acc(results); print(f"\n  Overall: {ov:.2f} (n={len(results)})")
    return {"benchmark":"RealWorldQA","overall":ov,"n_total":len(results)}, results

def concat_images_horizontal(paths, max_h=768):
    if len(paths)==1: return paths[0]
    imgs=[Image.open(p).convert("RGB") for p in paths]
    th=min(max_h,max(i.height for i in imgs))
    rs=[i.resize((int(i.width*th/i.height),th),Image.LANCZOS) for i in imgs]
    tw=sum(i.width for i in rs); c=Image.new("RGB",(tw,th)); x=0
    for i in rs: c.paste(i,(x,0)); x+=i.width
    c.save("/tmp/ssl_concat.png"); return "/tmp/ssl_concat.png"

def eval_blink(model, processor, subtasks=None, max_samples=None):
    print(f"\n{'='*60}\n  BLINK (SSL)\n{'='*60}")
    if subtasks is None: subtasks=ALL_BLINK_SUBTASKS
    items=[]
    for st in subtasks:
        try:
            ds=load_dataset("BLINK-Benchmark/BLINK",st,split="val")
            for i,it in enumerate(ds):
                ch=it.get("choices",it.get("options",[]));
                if isinstance(ch,str):
                    try: ch=json.loads(ch)
                    except: ch=[c.strip() for c in ch.split(",")]
                items.append({"idx":f"{st}_{i}","subtask":st,"prompt":it.get("prompt",it.get("question","")),"answer":it.get("answer",""),"choices":ch,"image_1":it.get("image_1",it.get("image",None)),"image_2":it.get("image_2",None),"image_3":it.get("image_3",None),"image_4":it.get("image_4",None)})
            print(f"  {st}: {len(ds)}")
        except Exception as e: print(f"  [error] {st}: {e}")
    if max_samples and len(items)>max_samples:
        pt=max(1,max_samples//len(subtasks)); bt=defaultdict(list)
        for it in items: bt[it["subtask"]].append(it)
        s=[]; [s.extend(its[:pt]) for its in bt.values()]; items=s[:max_samples]
    print(f"  Total: {len(items)}")
    results=[]; td="/tmp/ssl_blink"; os.makedirs(td,exist_ok=True)
    for item in tqdm(items,desc="BLINK"):
        ch=item["choices"]; hic=isinstance(ch,list) and len(ch)>0 and isinstance(ch[0],Image.Image)
        if hic: ct="\n".join([f"{chr(ord('A')+i)}. (Image {i+1})" for i in range(len(ch))])
        else: ct="\n".join([f"{chr(ord('A')+i)}. {c}" for i,c in enumerate(ch)])
        prompt=f"Answer the following question.\nSelect the correct option and output ONLY the letter.\nDo NOT output explanation.\n\n{item['prompt']}\n\nChoices:\n{ct}\n\nAnswer:"
        ips=[]
        for key in ["image_1","image_2","image_3","image_4"]:
            im=item.get(key)
            if im is not None and isinstance(im,Image.Image): p=os.path.join(td,f"{key}.png"); im.save(p); ips.append(p)
        if not ips: results.append({"idx":item["idx"],"subtask":item["subtask"],"answer":item["answer"],"response":"","extracted":None,"correct":False}); continue
        cp=concat_images_horizontal(ips); resp=qwen_generate(model,processor,cp,prompt)
        ext=extract_choice_letter(resp); ans=item["answer"]; ef=f"({ext})" if ext else None
        results.append({"idx":item["idx"],"subtask":item["subtask"],"answer":ans,"response":resp,"extracted":ef,"correct":ef==ans})

    bs=defaultdict(list)
    for r in results: bs[r["subtask"]].append(r)
    def acc(it): return sum(1 for i in it if i["correct"])/len(it)*100 if it else 0
    ps={t:acc(bs[t]) for t in sorted(bs)}; ov=sum(ps.values())/len(ps) if ps else 0
    di=[r for t in DEPTH_SPATIAL_SUBTASKS for r in bs.get(t,[])]; ad=acc(di)
    nu=sum(1 for r in results if r["extracted"] is None)
    print(f"\n  Overall: {ov:.2f}  DS: {ad:.2f}  Unparsed: {nu}/{len(results)}")
    return {"benchmark":"BLINK","overall":ov,"acc_depth_spatial":ad,"per_subtask":ps,"n_unparsed":nu,"n_total":len(results)}, results


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser=argparse.ArgumentParser(description="SSL (EMNLP 2025) on Qwen2.5-VL")
    parser.add_argument("--mode",type=str,default="both",choices=["identify","eval","both"])
    parser.add_argument("--benchmarks",type=str,nargs="+",default=["cvbench","mmstar","realworldqa","blink"],choices=["cvbench","mmstar","realworldqa","blink"])
    parser.add_argument("--model_id",type=str,default=CFG.model_id)
    parser.add_argument("--sae_ckpt_dir",type=str,default=CFG.save_dir)
    parser.add_argument("--layer",type=int,default=CFG.vis_layer)
    parser.add_argument("--latent_mult",type=int,default=CFG.latent_mult)
    parser.add_argument("--topk",type=int,default=CFG.topk)
    parser.add_argument("--gamma",type=float,default=0.2,help="SSL steering strength")
    parser.add_argument("--identify_samples",type=int,default=500)
    parser.add_argument("--max_samples",type=int,default=None)
    parser.add_argument("--save_dir",type=str,default="outputs/ssl_qwen25vl_results")
    parser.add_argument("--features_path",type=str,default=None)
    parser.add_argument("--blink_subtasks",type=str,nargs="+",default=None)
    parser.add_argument("--depth_spatial_only",action="store_true")
    args=parser.parse_args()
    os.makedirs(args.save_dir,exist_ok=True)

    model, processor, sae = load_model_and_sae(args.model_id, args.sae_ckpt_dir, args.layer, args.latent_mult, args.topk)

    fp = args.features_path or os.path.join(args.save_dir, f"ssl_features_layer{args.layer}.pt")

    ssl_steer = None
    if args.mode in ["identify","both"]:
        ssl_steer = identify_ssl_features(model, processor, sae, args.layer, args.identify_samples, fp)
        if ssl_steer is None: print("ERROR: Identification failed!"); return

    if args.mode in ["eval","both"]:
        if ssl_steer is None:
            ssl_steer = SSLSteering(model, sae, processor=processor, steer_layer=args.layer, gamma=args.gamma)
            ssl_steer.load_features(fp)
        else:
            ssl_steer.gamma = args.gamma
        ssl_steer.patch_generate()
        print(f"\n  SSL active: gamma={args.gamma}, layer={args.layer}")

        config={"method":"SSL","model_id":args.model_id,"gamma":args.gamma,"steer_layer":args.layer,"faithful_idx":ssl_steer.faithful_idx,"hall_idx":ssl_steer.hall_idx}
        with open(os.path.join(args.save_dir,"config.json"),"w") as f: json.dump(config,f,indent=2)

        summary={}
        if "cvbench" in args.benchmarks:
            m,r=eval_cvbench(model,processor,args.max_samples); summary["cvbench"]=m
            with open(os.path.join(args.save_dir,"cvbench_results.json"),"w") as f: json.dump(r,f,indent=2,ensure_ascii=False)
        if "mmstar" in args.benchmarks:
            m,r=eval_mmstar(model,processor,args.max_samples); summary["mmstar"]=m
            with open(os.path.join(args.save_dir,"mmstar_results.json"),"w") as f: json.dump(r,f,indent=2,ensure_ascii=False)
        if "realworldqa" in args.benchmarks:
            m,r=eval_realworldqa(model,processor,args.max_samples); summary["realworldqa"]=m
            with open(os.path.join(args.save_dir,"realworldqa_results.json"),"w") as f: json.dump(r,f,indent=2,ensure_ascii=False)
        if "blink" in args.benchmarks:
            st=args.blink_subtasks; 
            if args.depth_spatial_only: st=DEPTH_SPATIAL_SUBTASKS
            m,r=eval_blink(model,processor,st,args.max_samples); summary["blink"]=m
            with open(os.path.join(args.save_dir,"blink_results.json"),"w") as f: json.dump(r,f,indent=2,ensure_ascii=False)

        print(f"\n{'='*60}\n  SSL (gamma={args.gamma}) — Summary\n{'='*60}")
        for n,m in summary.items():
            k="cv_bench" if n=="cvbench" else "overall"; print(f"  {n:15s}: {m.get(k,0):.2f}")
        print(f"{'='*60}")
        with open(os.path.join(args.save_dir,"summary.json"),"w") as f: json.dump(summary,f,indent=2)
        print(f"\nDone. Results saved to {args.save_dir}/")

if __name__=="__main__": main()