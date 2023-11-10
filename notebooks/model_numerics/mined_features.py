# %%
import torch
import pickle

subject = "k"
face_parts = ["lips", "skin", "eyes", "nose"]

with open("misc/L14_real_semantic_patches_v2_2000.pickle", "rb") as f:
    data = pickle.load(f)

layers = len(data[subject][face_parts[0]])

for l in range(layers):
    features = torch.stack(
        [
            data[subject][part][l]
            for part in face_parts
        ]
    )
    print(
        f"{l}:\n",
        1 + torch.nn.functional.cosine_similarity(
            features.unsqueeze(1),
            features.unsqueeze(0),
            dim=-1
        )
    )

# %%
