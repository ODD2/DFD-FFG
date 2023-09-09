import torch
from src.clip import clip
from src.clip.model_vpt import PromptMode

x = torch.randn(5, 3, 224, 224).cuda()

with torch.no_grad():
    m, t = clip.load("ViT-B/16", prompt_mode=PromptMode.NONE, prompt_layers=0, prompt_num=0)
    m = m.visual.float().cuda()
    m(x)
    del m, t

for prompt_mode in [PromptMode.SHALLOW, PromptMode.DEEP, PromptMode.DEEPC]:
    m, t = clip.load("ViT-B/16", prompt_mode=prompt_mode, prompt_layers=3, prompt_num=10)
    m = m.visual.float().cuda()
    m.requires_grad_(False)
    for param in m.prompt_parameters():
        param.requires_grad_(True)
    m(x)
    del m, t
