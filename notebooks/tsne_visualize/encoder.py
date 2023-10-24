# %%
import os
import cv2
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from deepface import DeepFace
from sklearn.manifold import TSNE
from notebooks.tools import load_model
from src.model.clip.ftfe import LinearMeanVideoLearner
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from src.dataset.ffpp import FFPP, FFPPAugmentation, FFPPSampleStrategy

DEVICE = "cuda"


def tsne(
    embeddings,
    labels,
    perplexities=[10, 50, 80],
    graph_title="",
    save_path="",
    save=False
):
    X = np.array(embeddings)
    plt.figure(figsize=(5 * len(perplexities), 4), layout="constrained")
    plt.suptitle(graph_title)
    for i, perplexity in enumerate(perplexities):
        print(f"TSNE Running(Perplexity:{perplexity})....")
        tsne = TSNE(
            n_components=2,
            n_iter=10000,
            learning_rate='auto',
            init='random',
            perplexity=perplexity,
            # metric="cosine"
        )
        _X = tsne.fit_transform(X)
        print(f"TSNE Completed.")

        color = {
            "REAL": "green",
            "DF": "blue",
            "FS": "purple",
            "F2F": "darkorange",
            "NT": "red",
            "CDF_REAL": "turquoise",
            "CDF_FAKE": "deeppink",
            "DFDC_REAL": "forestgreen",
            "DFDC_FAKE": "steelblue",
            "Woman": "red",
            "Man": "blue"
        }

        plt.subplot(1, len(perplexities), i + 1)
        plt.title(f"Perplexity:{perplexity}")
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

        offset = 0
        for category_type, num in labels:
            plt.scatter(
                _X[offset:offset + num, 0],
                _X[offset:offset + num, 1],
                3,
                # color="green" if category_type == "REAL" else "red",
                color=color[category_type],
                label=category_type if i == 0 else "",
                alpha=0.5
            )
            offset += num

        assert offset == len(_X)

    plt.gcf().legend(loc='outside right center')
    if (save):
        folder, file = os.path.split(save_path)
        if (not os.path.exists(folder)):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


# %%
NUM_SAMPLES = 50


model_configs = [
    ("pre", LinearMeanVideoLearner, ""),
    (0, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=0-step=63.ckpt"),
    (1, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=1-step=126.ckpt"),
    (2, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=2-step=189.ckpt"),
    (3, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=3-step=252.ckpt"),
    (4, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=4-step=315.ckpt"),
    (5, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=5-step=378.ckpt"),
    (10, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=10-step=693.ckpt"),
    (15, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=15-step=1008.ckpt"),
    (20, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=20-step=1323.ckpt"),
    (29, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=29-step=1890.ckpt"),
    # (532, LinearMeanVideoLearner, "logs/DFD-FFG/unquhc4n/checkpoints/epoch=49-step=10650.ckpt"),
    # (534, LinearMeanVideoLearner, "logs/DFD-FFG/4459847z/checkpoints/epoch=23-step=5112.ckpt")
]


@torch.no_grad()
def runner(model_config, timestr):
    model_name, model_cls, model_path = model_config
    model = load_model(model_cls, model_path).model
    model.to(DEVICE)
    model.eval()
    aggregate_embeddings = {
        "Woman": [],
        "Man": []
    }
    anc_data = {
        "probs": [],
        "preds": [],
        "labels": []
    }
    for df_type in ["REAL", "DF", "FS", "NT", "F2F"]:
        dataset = FFPP(
            df_types=[df_type],
            compressions=["c23"],
            n_px=model.n_px,
            strategy=FFPPSampleStrategy.NORMAL,
            augmentations=FFPPAugmentation.NONE,
            force_random_speed=False,
            vid_ext=".avi",
            data_dir="datasets/ffpp/",
            num_frames=1,
            clip_duration=4,
            split="val",
            transform=model.transform,
            pack=False,
            ratio=1.0
        )

        random.seed(1234)

        # random sample datas
        clips = []
        for idx in [random.randrange(0, len(dataset)) for _ in range(NUM_SAMPLES)]:
            clips.append(dataset.get_entity(idx)["clips"])
        clips = torch.cat(clips, dim=0).to(DEVICE)

        # detect gender
        gender_clip_idx_map = {
            "Woman": [],
            "Man": []
        }
        for clip_idx in range(clips.shape[0]):
            image = clips[[clip_idx]].cpu()
            image = image.flatten(0, 2).permute(1, 2, 0).numpy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gender = DeepFace.analyze(
                img_path=image,
                actions=['gender'],
                enforce_detection=False,
                silent=True
            )[0]["dominant_gender"]
            gender_clip_idx_map[gender].append(clip_idx)

        # extract features (batched)
        results = model(clips)
        logits = results["logits"].detach().cpu()
        embeds = results["embeds"].mean(1).detach().cpu()

        # record embeddings
        # aggregate_embeddings[df_type] = embeds
        for gender, clip_indices in gender_clip_idx_map.items():
            aggregate_embeddings[gender].append(embeds[clip_indices])

        # check integrity
        probs = logits.softmax(dim=-1)[:, 1].numpy()
        preds = (probs > 0.5).astype(np.int8)
        labels = [0 if df_type == "REAL" else 1] * probs.shape[0]
        anc_data["probs"] += probs.flatten().tolist()
        anc_data["preds"] += preds.flatten().tolist()
        anc_data["labels"] += labels

    acc = round(accuracy_score(anc_data["labels"], anc_data["preds"]), 3)
    roc = round(roc_auc_score(anc_data["labels"], anc_data["probs"]), 3)
    mat = confusion_matrix(anc_data["labels"], anc_data["preds"])

    features = []
    labels = []
    for k, l in aggregate_embeddings.items():
        labels.append((k, sum([len(_l) for _l in l])))
        features.extend(l)
    features = torch.cat(features, dim=0)

    tsne(
        features,
        labels,
        perplexities=[2, 5, 10, 50],
        graph_title=f"{model_name}/acc:{acc}/auc:{roc}/tn:{mat[0,0]}/tp:{mat[1,1]}/fn:{mat[1,0]}/fp:{mat[0,1]}",
        save_path=f"./misc/extern/tsne/{timestr}-{model_name}.pdf",
        save=True,
    )


import gc
timestr = time.strftime("%H%M%S")
for m in model_configs:
    runner(m, timestr)
    gc.collect()
    torch.cuda.empty_cache()


# %%
