# %%
import torch
from src.clip import clip as CLIP
from src.model.clip.ftfe import LinearMeanVideoLearner
from src.clip.model_vpt import PromptMode

DEVICE = "cuda"
LOAD_MODEL_PARAM = (LinearMeanVideoLearner, "logs/DFD-FFG/4xn6o6wv/checkpoints/epoch=9-step=801.ckpt")
NUM_FRAME_SAMPLE = 1


def load_model(model_cls, ckpt_path):
    model = model_cls.load_from_checkpoint(ckpt_path, strict=False, attn_record=True)
    model.to(DEVICE)
    model.eval()
    return model


# %%
model = load_model(*LOAD_MODEL_PARAM)
model = model.to(DEVICE)
vptune = model.model.encoder.model
assert vptune.transformer.prompt_mode == PromptMode.DEEPC

# %%


def hook(module, input):
    patchs = input[0]
    tokens = patchs.shape[0] - module.prompt_num
    module.in_prompt = patchs[tokens:]


for blk in vptune.transformer.resblocks:
    blk.register_forward_pre_hook(hook)

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
frames = torch.cat(
    [
        dataset.get_entity(
            random.randrange(0, len(dataset)),
            with_entity_info=True
        )["clips"] for _ in range(NUM_FRAME_SAMPLE)
    ],
    dim=0
).flatten(0, 1).to(DEVICE)

# %%
with torch.no_grad():
    vptune(frames)

# %%
layer_prompt_record = []
for blk in vptune.transformer.resblocks:
    if (blk.prompt_mode == PromptMode.NONE):
        continue
    prompt_impact_ratio = (blk.prompts / blk.in_prompt).abs()
    layer_prompt_record.append(
        {
            "raw": blk.in_prompt,
            "add": blk.prompts
        }
    )

# %%
layer_prompt_record[0][0].shape
# %%
len(layer_prompt_record)

# %%
