# %%
import torch


def extract_encoder_from_checkpoint(weight_path, encoder_path="model.encoder.model"):
    data = torch.load(weight_path, map_location="cpu")
    state_dict = {k[len(encoder_path) + 1:]: v for k, v in data["state_dict"].items() if encoder_path in k}
    torch.save(state_dict, weight_path.replace(".ckpt", "_encoder.pth"))


extract_encoder_from_checkpoint("logs/DFD-FFG/71hfy89x/checkpoints/epoch=38-step=8307.ckpt")

# %%
