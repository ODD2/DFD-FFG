# %%
import time
import random
import torchvision
from torchvision.io import VideoReader

# %%
vid_reader = VideoReader(
    "datasets/ffpp/DF/c23/videos/000_003.avi",
    "video",
    num_threads=0
)
# %%
duration = vid_reader.get_metadata()["video"]["duration"][0]

t = time.time()
num_iters = 4
for _ in range(num_iters):
    for _ in range(100):
        vid_reader.seek((duration * random.random()) - 0.1)
        next(vid_reader)

print((time.time() - t) / num_iters)
del vid_reader

# %%
