# %%
import torch
# import clip as CLIP
from src.clip import clip as CLIP
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = CLIP.load("ViT-B/16", device=device)
model = model.float()

# %%
image = Image.open("man.png")
image = preprocess(image).unsqueeze(0).to(device)

# %%
text = CLIP.tokenize(["a man", "a woman"]).to(device)
# %%

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
# %%
from src.model.clip import FrameAttrExtractor
from src.clip.model_vpt import PromptMode
image_embed = FrameAttrExtractor("ViT-B/16", PromptMode.NONE, 0, 0, 0, True)
image_embed = image_embed.to(device)
# %%
from src.model.clip.ftfe import TextMeanVideoLearner

text_embed = TextMeanVideoLearner.load_from_checkpoint("logs/DFD-FFG/y3edez8n/checkpoints/last.ckpt", "cpu").model.proj
text_embed = text_embed.to(device)
# %%
with torch.no_grad():
    _image_features = image_embed(image)["embeds"]
    logits = text_embed(_image_features, mean_mode="none")

    probs = logits.softmax(dim=-1).cpu().numpy()
probs
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
# %%
# %%
