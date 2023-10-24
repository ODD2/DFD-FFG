import torch
# import clip as CLIP
from src.clip import clip as CLIP
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = CLIP.load("ViT-B/16", device=device)
model = model.float()
image = Image.open("woman.png")
image = preprocess(image).unsqueeze(0).to(device)
text = CLIP.tokenize(["a man", "a woman"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
