import os
import shutil
import torch
from safetensors.torch import load_file, save_file

SRC_DIR = "./trained_models/xTransform/qwen3_8B_cyber_actmap_v4_neg"
DST_DIR = "./trained_models/xTransform/qwen3_8B_cyber_actmap_v4_neg_nlpow15"
GAMMA = 1.5
EPS = 1e-12

os.makedirs(DST_DIR, exist_ok=True)
for fname in os.listdir(SRC_DIR):
    if fname != "adapter_model.safetensors":
        shutil.copy2(os.path.join(SRC_DIR, fname), os.path.join(DST_DIR, fname))

state = load_file(os.path.join(SRC_DIR, "adapter_model.safetensors"))
new_state = {}
changed = 0

for k, v in state.items():
    if ".lora_B." in k:
        t = v.float()
        # Data-free nonlinear remap: signed power, then per-tensor norm preserve.
        t_new = torch.sign(t) * torch.pow(torch.abs(t) + EPS, GAMMA)
        n_old = torch.norm(t)
        n_new = torch.norm(t_new)
        if n_new > EPS and n_old > EPS:
            t_new = t_new * (n_old / n_new)
        new_state[k] = t_new.to(v.dtype)
        changed += 1
    else:
        new_state[k] = v

save_file(new_state, os.path.join(DST_DIR, "adapter_model.safetensors"))
print(f"Saved nonlinear remapped adapter to {DST_DIR}")
print(f"Changed lora_B tensors: {changed}")
