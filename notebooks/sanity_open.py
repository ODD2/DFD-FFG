import torch
from PIL import Image
import open_clip
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
tokenizer = open_clip.get_tokenizer('ViT-L-14')
model.eval()

image = Image.open("notebooks/woman.png")
image = preprocess(image).unsqueeze(0)
text = tokenizer(["a man", "a woman"])
print("load complete, start evaluation...")
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", probs)  # Label probs: [[0.03804807 0.96195185]]
