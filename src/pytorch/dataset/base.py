import torchvision
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms

torchvision.set_video_backend("video_reader")
