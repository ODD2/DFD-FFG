import cv2
import json
import torch
import pickle
import random
import torchvision
import numpy as np
import torch.nn as nn
import albumentations as alb
import lightning.pytorch as pl

from tqdm import tqdm
from functools import partial
from enum import IntEnum, auto, IntFlag, Enum
from os import path, scandir, makedirs
from torchvision.io import VideoReader
from torch.utils.data import Dataset, DataLoader
from typing import List, Set, Dict, Tuple, Optional, Callable, Union


torchvision.set_video_backend("video_reader")
