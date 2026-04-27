"""
Dense Cross-Attention Baseline 评估脚本 (支持 num_runs + mean±std)。

用法：
    python src_ablation/eval_dense_xattn_new.py --predictor_ckpt outputs/dense_xattn_ckpt/predictor_best.pt --benchmarks cvbench blink --num_runs 3 --subsample_ratio 0.8
"""
import os, sys, re, json, random, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
try:
    from config import CFG
except ImportError:
    class CFG:
        model_id="Qwen/Qwen2.5-VL-7B-Instruct"; vis_layer=8; top_n_patches=60; device="cuda"

ALL_BLINK_SUBTASKS=["Art_Style","Counting","Forensic_Detection","Functional_Correspondence","IQ_Test","Jigsaw","Multi-view_Reasoning","Object_Localization","Relative_Depth","Relative_Reflectance","Semantic_Correspondence","Spatial_Relation","Visual_Correspondence","Visual_Similarity"]
DEPTH_SPATIAL_SUBTASKS=["Relative_Depth","Spatial_Relation","Multi-view_Reasoning","Object_Localization"]

def extract_choice_letter(response, max_letter="Z"):
    text=response.strip().split("\n")[0].strip().upper()
    if not text: return None
    pat=f'[A-{max_letter}]'
    m=re.search(rf'\(({pat})\)',text)
    if m: return f"({m.group(1)})"
    m=re.search(rf'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?({pat})\)?',text)
    if m: return f"({m.group(1)})"
    m=re.match(rf'^({pat})(?:[\s.,):]|$)',text)
    if m: return f"({m.group(1)})"
    return None

def match_by_content(response, choices):
    resp=response.strip().split("\n")[0].strip().lower()
    for i,c in enumerate(choices):
        if isinstance(c,str) and resp==c.strip().lower(): return f"({chr(ord('A')+i)})"
    return None

def concat_images_horizontal(paths, max_h=768):
    if len(paths)==1: return paths[0]
    imgs=[Image.open(p).convert("RGB") for p in paths]
    th=min(max_h,max(i.height for i in imgs))
    rs=[i.resize((int(i.width*th/i.height),th),Image.LANCZOS) for i in imgs]
    tw=sum(i.width for i in rs); c=Image.new("RGB",(tw,th)); x=0
    for i in rs: c.paste(i,(x,0)); x+=i.width
    c.save("/tmp/dx_concat.png"); return "/tmp/dx_concat.png"

def subsample(ds, ratio, seed, is_list=False):
    random.seed(seed); n=len(ds); k=int(n*ratio); idx=random.sample(range(n),k)
    return [ds[i] for i in idx] if is_list else ds.select(idx)

def load_model(args):
    from Model_dense_xattn import QwenWithDenseCrossAttention
    print("Loading Dense Cross-Attention baseline...")
    m=QwenWithDenseCrossAttention.from_pretrained(model_id=args.model_id,inject_layer=args.layer,top_n_patches=args.top_n_patches,predictor_ckpt=args.predictor_ckpt,device=args.device)
    m.to(args.device); m.eval(); return m

def _gen(model, img, prompt):
    tmp="/tmp/dx_tmp.png"
    if isinstance(img,Image.Image): img.save(tmp)
    else: tmp=img
    try:
        r=model.generate(image_path=tmp,question=prompt,verbose=False); return r["final_answer"]
    except Exception as e:
        print(f"  [error]: {e}"); return ""

def eval_cvbench(model, ds, label=""):
    print(f"\n  Evaluating CV-Bench: {label}")
    results=[]
    for item in tqdm(ds,desc=f"CV-Bench [{label}]"):
        choices=item["choices"]
        ct="\n".join([f"{chr(ord('A')+i)}. {c}" for i,c in enumerate(choices)])
        prompt=f"Answer the following question.\nSelect the correct option and output ONLY the letter.\nDo NOT output explanation.\n\n{item['prompt']}\n\nChoices:\n{ct}\n\nAnswer:"
        resp=_gen(model,item["image"],prompt)
        ext=extract_choice_letter(resp)
        if ext is None: ext=match_by_content(resp,choices)
        ans=item["answer"]; correct=ext==ans
        results.append({"idx":item["idx"],"task":item["task"],"source":item["source"],"type":item["type"],"answer":ans,"response":resp,"extracted":ext,"correct":correct})
    return results

def compute_cvbench_metrics(results, label=""):
    bs=defaultdict(list); bt=defaultdict(list)
    for r in results: bs[r["source"]].append(r); bt[r["task"]].append(r)
    def acc(it): return sum(1 for i in it if i["correct"])/len(it)*100 if it else 0
    aa=acc(bs.get("ADE20K",[])); ac=acc(bs.get("COCO",[])); ao=acc(bs.get("Omni3D",[]))
    a2=(aa+ac)/2; a3=ao; cv=(a2+a3)/2; nu=sum(1 for r in results if r["extracted"] is None)
    print(f"\n  CV-Bench [{label}]: Overall={cv:.2f} 2D={a2:.2f} 3D={a3:.2f} Unparsed={nu}/{len(results)}")
    pt={t:acc(it) for t,it in bt.items()}
    for t in sorted(pt): print(f"    {t:20s}: {pt[t]:6.2f} (n={len(bt[t])})")
    return {"cv_bench":cv,"acc_2d":a2,"acc_3d":a3,"acc_ade":aa,"acc_coco":ac,"acc_omni":ao,"per_task":pt,"n_unparsed":nu,"n_total":len(results)}

def eval_mmstar(model, ds, label=""):
    print(f"\n  Evaluating MMStar: {label}")
    results=[]
    for item in tqdm(ds,desc=f"MMStar [{label}]"):
        prompt=f"Answer the following question.\nSelect the correct option and output ONLY the letter (A, B, C, or D).\nDo NOT output explanation.\n\n{item['question']}\n\nAnswer:"
        resp=_gen(model,item["image"],prompt)
        ext=extract_choice_letter(resp,max_letter="D")
        el=ext.strip("()") if ext else None; ans=item["answer"].strip().upper(); correct=el==ans
        results.append({"index":item["index"],"category":item["category"],"l2_category":item["l2_category"],"answer":ans,"response":resp,"extracted":ext,"correct":correct})
    return results

def compute_mmstar_metrics(results, label=""):
    def acc(it): return sum(1 for i in it if i["correct"])/len(it)*100 if it else 0
    ov=acc(results); nu=sum(1 for r in results if r["extracted"] is None)
    print(f"\n  MMStar [{label}]: Overall={ov:.2f} Unparsed={nu}/{len(results)}")
    return {"overall":ov,"n_unparsed":nu,"n_total":len(results)}

def eval_realworldqa(model, ds, label=""):
    print(f"\n  Evaluating RealWorldQA: {label}")
    results=[]
    for i,item in enumerate(tqdm(ds,desc=f"RealWorldQA [{label}]")):
        q=item["question"]; a=item["answer"]; resp=_gen(model,item["image"],q)
        ans=a.strip()
        if ans in ("A","B","C","D"):
            ext=extract_choice_letter(resp,max_letter="D"); el=ext.strip("()") if ext else None; correct=el==ans
        elif ans.lower() in ("yes","no"):
            rl=resp.strip().lower(); correct=(rl.startswith("yes") and ans.lower()=="yes") or (rl.startswith("no") and ans.lower()=="no"); ext=resp[:50]
        else:
            rn=resp.strip().split("\n")[0].strip().lower().rstrip("."); an=ans.lower().rstrip(".")
            correct=rn==an or rn.startswith(an); ext=resp[:50]
        results.append({"idx":i,"answer":a,"response":resp,"extracted":ext,"correct":correct})
    return results

def compute_realworldqa_metrics(results, label=""):
    def acc(it): return sum(1 for i in it if i["correct"])/len(it)*100 if it else 0
    ov=acc(results); print(f"\n  RealWorldQA [{label}]: Overall={ov:.2f} (n={len(results)})"); return {"overall":ov,"n_total":len(results)}

def load_blink(subtasks, max_samples=None):
    print("Loading BLINK...")
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
            print(f"  {st}: {len(ds)} samples")
        except Exception as e: print(f"  [error] {st}: {e}")
    if max_samples and len(items)>max_samples:
        pt=max(1,max_samples//len(subtasks)); bt=defaultdict(list)
        for it in items: bt[it["subtask"]].append(it)
        s=[];
        for its in bt.values(): s.extend(its[:pt])
        items=s[:max_samples]
    print(f"  Total: {len(items)} samples"); return items

def eval_blink(model, dataset, label=""):
    print(f"\n  Evaluating BLINK: {label}")
    td="/tmp/dx_blink"; os.makedirs(td,exist_ok=True); results=[]
    for item in tqdm(dataset,desc=f"BLINK [{label}]"):
        ch=item["choices"]
        hic=isinstance(ch,list) and len(ch)>0 and isinstance(ch[0],Image.Image)
        if hic: ct="\n".join([f"{chr(ord('A')+i)}. (Image {i+1})" for i in range(len(ch))])
        else: ct="\n".join([f"{chr(ord('A')+i)}. {c}" for i,c in enumerate(ch)])
        prompt=f"Answer the following question.\nSelect the correct option and output ONLY the letter.\nDo NOT output explanation.\n\n{item['prompt']}\n\nChoices:\n{ct}\n\nAnswer:"
        ips=[]
        for key in ["image_1","image_2","image_3","image_4"]:
            im=item.get(key)
            if im is not None and isinstance(im,Image.Image):
                p=os.path.join(td,f"{key}.png"); im.save(p); ips.append(p)
        if not ips: results.append({"idx":item["idx"],"subtask":item["subtask"],"answer":item["answer"],"response":"","extracted":None,"correct":False}); continue
        cp=concat_images_horizontal(ips)
        resp=_gen(model,cp,prompt)
        ext=extract_choice_letter(resp)
        if ext is None:
            cm=match_by_content(resp,ch)
            if cm: ext=cm
        ans=item["answer"]; correct=ext==ans
        results.append({"idx":item["idx"],"subtask":item["subtask"],"answer":ans,"response":resp,"extracted":ext,"correct":correct})
    return results

def compute_blink_metrics(results, label=""):
    bs=defaultdict(list)
    for r in results: bs[r["subtask"]].append(r)
    def acc(it): return sum(1 for i in it if i["correct"])/len(it)*100 if it else 0
    ps={t:acc(bs[t]) for t in sorted(bs)}; ov=sum(ps.values())/len(ps) if ps else 0
    di=[r for t in DEPTH_SPATIAL_SUBTASKS for r in bs.get(t,[])]
    oi=[r for t in bs if t not in DEPTH_SPATIAL_SUBTASKS for r in bs[t]]
    ad=acc(di); ao=acc(oi); nu=sum(1 for r in results if r["extracted"] is None)
    print(f"\n  BLINK [{label}]: Overall={ov:.2f} DS={ad:.2f} Other={ao:.2f} Unparsed={nu}/{len(results)}")
    return {"overall":ov,"acc_depth_spatial":ad,"acc_other":ao,"per_subtask":ps,"n_unparsed":nu,"n_total":len(results)}

def aggregate_multi_runs(runs):
    benchmarks=set()
    for r in runs: benchmarks.update(r.keys())
    agg={}
    for b in benchmarks:
        ms=[r[b] for r in runs if b in r]
        if not ms: continue
        a={}
        for k in ms[0]:
            vs=[m[k] for m in ms if k in m]
            if vs and isinstance(vs[0],(int,float)): a[k]={"mean":float(np.mean(vs)),"std":float(np.std(vs)),"runs":[float(v) for v in vs]}
        agg[b]=a
    return agg

def print_summary_with_std(agg):
    print(f"\n{'═'*70}\n  Dense Cross-Attention Baseline — Summary (mean ± std)\n{'═'*70}")
    for name,bench,key in [("CV-Bench Overall","cvbench","cv_bench"),("  2D","cvbench","acc_2d"),("  3D","cvbench","acc_3d"),("MMStar","mmstar","overall"),("RealWorldQA","realworldqa","overall"),("BLINK Overall","blink","overall"),("  Depth/Spatial","blink","acc_depth_spatial")]:
        if bench not in agg: continue
        s=agg[bench].get(key)
        if s: print(f"  {name:<25s}: {s['mean']:6.2f} ± {s['std']:.2f}")
    print(f"{'═'*70}")

def main():
    parser=argparse.ArgumentParser(description="Dense XAttn Baseline Eval")
    parser.add_argument("--benchmarks",type=str,nargs="+",default=["cvbench","mmstar","realworldqa","blink"],choices=["cvbench","mmstar","realworldqa","blink"])
    parser.add_argument("--model_id",type=str,default=CFG.model_id)
    parser.add_argument("--layer",type=int,default=CFG.vis_layer)
    parser.add_argument("--top_n_patches",type=int,default=CFG.top_n_patches)
    parser.add_argument("--predictor_ckpt",type=str,default="outputs/dense_xattn_ckpt/predictor_best.pt")
    parser.add_argument("--save_dir",type=str,default="outputs/dense_xattn_results")
    parser.add_argument("--max_samples",type=int,default=None)
    parser.add_argument("--device",type=str,default=CFG.device)
    parser.add_argument("--num_runs",type=int,default=1)
    parser.add_argument("--subsample_ratio",type=float,default=0.8)
    parser.add_argument("--blink_subtasks",type=str,nargs="+",default=None)
    parser.add_argument("--depth_spatial_only",action="store_true")
    args=parser.parse_args()
    os.makedirs(args.save_dir,exist_ok=True)

    model=load_model(args)

    df={}
    if "cvbench" in args.benchmarks:
        ds=load_dataset("nyu-visionx/CV-Bench",split="test")
        if args.max_samples: ds=ds.select(range(min(args.max_samples,len(ds))))
        df["cvbench"]=ds; print(f"  CV-Bench: {len(ds)} samples")
    if "mmstar" in args.benchmarks:
        ds=load_dataset("Lin-Chen/MMStar",split="val")
        if args.max_samples: ds=ds.select(range(min(args.max_samples,len(ds))))
        df["mmstar"]=ds; print(f"  MMStar: {len(ds)} samples")
    if "realworldqa" in args.benchmarks:
        ds=load_dataset("xai-org/RealworldQA",split="test")
        if args.max_samples: ds=ds.select(range(min(args.max_samples,len(ds))))
        df["realworldqa"]=ds; print(f"  RealWorldQA: {len(ds)} samples")
    if "blink" in args.benchmarks:
        st=args.blink_subtasks
        if args.depth_spatial_only: st=DEPTH_SPATIAL_SUBTASKS
        if st is None: st=ALL_BLINK_SUBTASKS
        df["blink"]=load_blink(st,args.max_samples)

    all_runs=[]
    for ri in range(args.num_runs):
        if args.num_runs>1: print(f"\n{'▶'*30} Run {ri+1}/{args.num_runs} (seed={42+ri}) {'◀'*30}\n")
        rsd=os.path.join(args.save_dir,f"run_{ri}") if args.num_runs>1 else args.save_dir
        os.makedirs(rsd,exist_ok=True); seed=42+ri; lb=f"run{ri}" if args.num_runs>1 else "dense_xattn"
        if args.num_runs>1:
            ds={k:subsample(v,args.subsample_ratio,seed,is_list=(k=="blink")) for k,v in df.items()}
            for k in ds: print(f"  Run {ri+1}: {k} sampled {len(ds[k])}/{len(df[k])}")
        else: ds=df
        rm={}
        if "cvbench" in ds:
            r=eval_cvbench(model,ds["cvbench"],lb); rm["cvbench"]=compute_cvbench_metrics(r,lb)
            with open(os.path.join(rsd,"cvbench_results.json"),"w") as f: json.dump(r,f,indent=2,ensure_ascii=False)
        if "mmstar" in ds:
            r=eval_mmstar(model,ds["mmstar"],lb); rm["mmstar"]=compute_mmstar_metrics(r,lb)
            with open(os.path.join(rsd,"mmstar_results.json"),"w") as f: json.dump(r,f,indent=2,ensure_ascii=False)
        if "realworldqa" in ds:
            r=eval_realworldqa(model,ds["realworldqa"],lb); rm["realworldqa"]=compute_realworldqa_metrics(r,lb)
            with open(os.path.join(rsd,"realworldqa_results.json"),"w") as f: json.dump(r,f,indent=2,ensure_ascii=False)
        if "blink" in ds:
            r=eval_blink(model,ds["blink"],lb); rm["blink"]=compute_blink_metrics(r,lb)
            with open(os.path.join(rsd,"blink_results.json"),"w") as f: json.dump(r,f,indent=2,ensure_ascii=False)
        all_runs.append(rm)

    if args.num_runs>1:
        agg=aggregate_multi_runs(all_runs); print_summary_with_std(agg)
        summary={"config":{"method":"Dense Cross-Attention Baseline","model_id":args.model_id,"layer":args.layer,"num_runs":args.num_runs,"subsample_ratio":args.subsample_ratio,"seeds":[42+i for i in range(args.num_runs)]},"aggregated":agg,"per_run":all_runs}
        with open(os.path.join(args.save_dir,"summary_mean_std.json"),"w") as f: json.dump(summary,f,indent=2)
        print(f"\nDone. Summary saved to {args.save_dir}/summary_mean_std.json")
    else:
        s=all_runs[0]
        print(f"\n{'═'*60}\n  Dense Cross-Attention Baseline — Summary\n{'═'*60}")
        for b,m in s.items():
            k="cv_bench" if b=="cvbench" else "overall"; print(f"  {b:15s}: {m.get(k,0):.2f}")
        print(f"{'═'*60}")
        with open(os.path.join(args.save_dir,"summary.json"),"w") as f: json.dump(s,f,indent=2)
        print(f"\nDone. Summary saved to {args.save_dir}/summary.json")

if __name__=="__main__": main()