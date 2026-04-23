import torch
state = torch.load("outputs/focus_ckpt_layer8_32/predictor_best.pt", map_location="cpu")
print(state["head.2.weight"].shape)