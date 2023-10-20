# %%
import torch
from src.clip import clip as CLIP
from src.model.clip.ftfe import LinearMeanVideoLearner

# This script compares the patch embedding similarity between prompt tuned ViT-B/14 and the original CLIP model.
# 1. Load the desire model by changing the model type and checkpoint path for the load_model function.
# 303: 10 prompts
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/x3ughk94/checkpoints/epoch=8-step=720.ckpt")
# 311: 100 prompts
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/4xn6o6wv/checkpoints/epoch=9-step=801.ckpt")
# 312: 200 prompts
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/s8ttt5x1/checkpoints/epoch=9-step=801.ckpt")
# 339: 100 prompts, 10 prompt layers
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/88h2cggt/checkpoints/epoch=9-step=777.ckpt")
# 340: 100 prompts, 2 prompt layers
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/uxri1cwv/checkpoints/epoch=9-step=801.ckpt")
# 343: 100 prompts, 6 prompt layers, lips, w = 10
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/nbv4975f/checkpoints/epoch=9-step=801.ckpt")
# 344: 100 prompts, 6 prompt layers, lips, w = 100
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/2cs6zm2l/checkpoints/epoch=9-step=753.ckpt")
# 345: 100 prompts, 6 prompt layers, lips, w = 1
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/e35bdhc0/checkpoints/epoch=9-step=801.ckpt")
# 347: 100 prompts, 6 prompt layers, lips+skin, w = 10
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/qt3rtuge/checkpoints/epoch=9-step=801.ckpt")
# 348: 100 prompts, 6 prompt layers, lips+skin+eye+nose, w = 10
# 353: 100 prompts, 6 prompt layers, v2, lips, w = 100
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/gk2q910g/checkpoints/epoch=9-step=801.ckpt")
# 355: 100 prompts, 6 prompt layers, v2(100s), lips, w = 100
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/37jq7sm9/checkpoints/epoch=9-step=753.ckpt")
# 357: 100 prompts, 6 prompt layers, v1+v2(100s), lips, w = 100
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/nwhaaott/checkpoints/epoch=9-step=801.ckpt")
# 359: 100 prompts, 6 prompt layers, v1(no<l4), lips, w = 10
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/b42bl1ew/checkpoints/epoch=9-step=753.ckpt")
# 360: 100 prompts, 10 prompt layers, v1(no<l4), lips, w = 10
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/mkt9ydu6/checkpoints/epoch=7-step=615.ckpt")
# 361: 100 prompts, 10 prompt layers, v1(no<l4), lips, w = 100
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/nlszpnd9/checkpoints/epoch=9-step=777.ckpt")
# 361: 100 prompts, 10 prompt layers, v2(no<l4), lips, w = 100
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/5tsw6bx5/checkpoints/epoch=9-step=777.ckpt")
# 367: 100 prompts, 10 prompt layers, v4(no<l4), lips, w = 100
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/dc93gfig/checkpoints/epoch=9-step=777.ckpt")
# 371
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/pyfpkz7x/checkpoints/epoch=7-step=615.ckpt")
# 388
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/0pp16j5h/checkpoints/epoch=64-step=1690.ckpt")
# 391
# LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/1fh5aot5/checkpoints/epoch=92-step=2418.ckpt")
# 408
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/u5cw7jt4/checkpoints/epoch=54-step=1430.ckpt")
# 419
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/d2jmjrkm/checkpoints/epoch=71-step=1872.ckpt")
# 421
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/w5lsp5gd/checkpoints/epoch=12-step=338.ckpt")
# 422
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/p6tmn0rs/checkpoints/epoch=46-step=1222.ckpt")
# 424
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/rk9yjd7k/checkpoints/epoch=33-step=884.ckpt")
# 425
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/e79593vw/checkpoints/epoch=53-step=1404.ckpt")
# 427
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/qv9wnhpq/checkpoints/epoch=27-step=728.ckpt")
# 432
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/u2thetze/checkpoints/epoch=53-step=1404.ckpt")
# 435
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/71l48qc8/checkpoints/epoch=29-step=780.ckpt")
# 442
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/luz2tpm1/checkpoints/epoch=59-step=1560.ckpt")
# 532
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/unquhc4n/checkpoints/epoch=49-step=10650.ckpt")
# 533
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/hk4xvo3q/checkpoints/epoch=33-step=7242.ckpt")
# 534
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/4459847z/checkpoints/epoch=17-step=3834.ckpt")

# %%
# 2. Change the number of samples
NUM_BATCHES = 30

BATCH_SIZE = 20

# 3. Change the VMAX value for visibility(0 <= VMAX <= 1)
VMAX = 0.4

# 4. Change the DEVICE to run at the device
DEVICE = "cuda"
# %%


def load_model(model_cls, ckpt_path):
    model = model_cls.load_from_checkpoint(ckpt_path, strict=False, attn_record=True)
    model.to(DEVICE)
    model.eval()
    return model


model = load_model(*LOAD_MODEL_PARAM)
vptune = model.model.encoder.model
base = CLIP.load("ViT-B/16")[0].visual.float().to(DEVICE)
# %%
from src.dataset.ffpp import FFPP, FFPPAugmentation, FFPPSampleStrategy
dataset = FFPP(
    df_types=["NT"],
    compressions=["c23"],
    n_px=model.n_px,
    strategy=FFPPSampleStrategy.NORMAL,
    augmentations=FFPPAugmentation.NONE,
    force_random_speed=False,
    vid_ext=".avi",
    data_dir="datasets/ffpp/",
    num_frames=1,
    clip_duration=4,
    split="train",
    transform=model.transform,
    pack=False,
    ratio=1.0
)
# %%
import random
from src.utility.visualize import dataset_entity_visualize
random.seed(1019)

with torch.no_grad():
    patch_attr_dis_sim_layer_map = [{} for _ in range(base.transformer.layers)]
    for _ in range(NUM_BATCHES):
        frames = torch.cat(
            [
                dataset.get_entity(
                    random.randrange(0, len(dataset)),
                    with_entity_info=True
                )["clips"] for _ in range(BATCH_SIZE)
            ],
            dim=0
        ).flatten(0, 1).to(DEVICE)

        vptune(frames)
        base(frames)

        for l, blk1, blk2 in zip(
            range(base.transformer.layers),
            vptune.transformer.resblocks,
            base.transformer.resblocks
        ):
            layer_attrs = []
            blk1_attrs = blk1.attn.get_attr()
            blk2_attrs = blk2.attn.get_attr()
            attr_dis_sim_map = {}
            for attr_name in ["q", "k", "v", "out"]:
                diff = (
                    1 - torch.nn.functional.cosine_similarity(
                        blk1_attrs[attr_name].flatten(2),
                        blk2_attrs[attr_name].flatten(2),
                        dim=-1
                    ).mean(dim=0)
                ) / 2 / NUM_BATCHES
                attr_dis_sim_map = diff[1:].view(14, 14).cpu().numpy()
                if (not attr_name in patch_attr_dis_sim_layer_map[l]):
                    patch_attr_dis_sim_layer_map[l][attr_name] = attr_dis_sim_map
                else:
                    patch_attr_dis_sim_layer_map[l][attr_name] += attr_dis_sim_map

# %%
import matplotlib.pyplot as plt
unit = 2
num_layers = len(patch_attr_dis_sim_layer_map)
num_attrs = len(patch_attr_dis_sim_layer_map[0].keys())
plt.figure(figsize=(unit * num_layers * 0.9, unit * num_attrs + 1), layout="constrained")
for y in range(num_layers):
    for x, attr in enumerate(patch_attr_dis_sim_layer_map[0].keys()):
        plt.subplot(num_attrs + 1, num_layers, x * num_layers + y + 1)
        plt.gca().axis('off')
        plt.imshow(
            patch_attr_dis_sim_layer_map[y][attr],
            vmin=0,
            vmax=VMAX
        )
        plt.title(f"{y}/{attr}")
plt.show()
plt.close()

# %%
