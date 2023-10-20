# %%
import notebooks.tools as tools
import torch
import torch.nn as nn
import lightning.pytorch as pl


from lightning.pytorch.tuner.tuning import Tuner
from torchvision import transforms
from torchvision.datasets import MNIST
from src.clip.clip import _to_tensor, _convert_image_to_rgb
from src.model.base import ODClassifier
from src.model.clip.snvl import CLIPBinaryVideoLearner, PromptMode
from src.model.clip.lprobe import CLIPBinaryLinearProb, LinearProbe
from src.dataset.ffpp import FFPPDataModule, FFPPAugmentation, FFPPSampleStrategy


torch.set_float32_matmul_precision('high')
IS_MNIST = False
BATCH_SIZE = 20

if IS_MNIST:
    CLASSES = 10

    def create_dtl(model):
        train_set = MNIST('./datasets/', download=True, train=True, transform=model.transform)
        return {"MNIST": torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)}

    class _TestModule:
        def training_step(self, batch, batch_idx):
            x, y = batch[list(batch.keys())[0]]
            logits = self.model(x)
            loss = nn.functional.cross_entropy(logits, y)
            return loss

else:
    CLASSES = 2
    FRAMES = 10

    def create_dtl(model):
        dtm = FFPPDataModule(
            ["REAL", "DF", "FS", "F2F", "NT"],
            ["c23"],
            data_dir="datasets/ffpp/",
            vid_ext='.avi',
            batch_size=BATCH_SIZE,
            num_workers=8,
            num_frames=FRAMES,
            clip_duration=4,
            force_random_speed=False,
            strategy=FFPPSampleStrategy.CONTRAST_RAND,
            augmentations=[
                FFPPAugmentation.NORMAL,
                FFPPAugmentation.VIDEO,
                FFPPAugmentation.VIDEO_RRC,
                FFPPAugmentation.FRAME
            ],
            pack=False,
            ratio=1.0,
            max_clips=3
        )

        dtm.prepare_data()
        dtm.affine_model(model)
        dtm.setup("fit")
        return {
            "FFPP": dtm.train_dataloader()
        }

    class _TestModule:
        pass


class TestModule(_TestModule):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.learning_rate = 0.001

    def train_dataloader(self):
        return create_dtl(self.model)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate
        )


class AgentBinaryLinear(nn.Module):
    def __init__(self, backbone="res"):
        super().__init__()
        if backbone == "res":
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            self.model.fc = nn.Identity()
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                _convert_image_to_rgb,
                _to_tensor,
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.dtype = torch.float32
        elif backbone == "clip":
            from src.clip import clip as CLIP
            # import clip as CLIP
            self.model, self.transform = CLIP.load("ViT-B/16", "cpu")
            self.model = self.model.visual.float()
            self.model.proj = None
            self.dtype = self.model.conv1.weight.dtype
            self.model.requires_grad_(False)
        elif backbone == "vit":
            import torchvision
            self.model = torchvision.models.vit_b_16(
                torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
            )
            self.model.heads = nn.Identity()
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                _convert_image_to_rgb,
                _to_tensor
            ])
            self.dtype = torch.float32
        self.linear = nn.Linear(768, CLASSES).type(self.dtype)

    @property
    def n_px(self):
        return 224

    def forward(self, x, *args, **kargs):
        if len(x.shape) > 4:
            b, t = x.shape[0:2]
            logits = self.linear(self.model(x.flatten(0, 1))).unflatten(0, (b, t)).mean(dim=1)
        else:
            logits = self.linear(self.model(x))
        return logits


class SNVL(TestModule, CLIPBinaryVideoLearner):
    pass


class LProbe(TestModule, CLIPBinaryLinearProb):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.model = LinearProbe(output_dim=CLASSES)


class Agent(TestModule, ODClassifier):
    def __init__(self, name="clip"):
        super().__init__()
        self.model = AgentBinaryLinear(name)


# model = TestModule(6, 10, 2, 'ViT-B/16', PromptMode.DEEPC, 10, 6, 0.2)
# model = TestModule(6, 10, 2)
# model = TestModule2()
# model = Agent('clip')
model = LProbe()
# model = SNVL(6, FRAMES, 2, 'ViT-B/16', PromptMode.DEEPC, 10, 6, 0.2)
model.configure_optimizers()
trainer = pl.Trainer(precision="32")
tuner = Tuner(trainer)

# Run learning rate finder
lr_finder = tuner.lr_find(model, mode='exponential', min_lr=1e-4, max_lr=0.5, early_stop_threshold=2)

# Results can be found in
print(lr_finder.results)
print(min(lr_finder.results["loss"]))

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()

new_lr = lr_finder.suggestion()
new_lr

# %%
