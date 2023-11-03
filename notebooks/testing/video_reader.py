import torchvision
from torchvision.io import VideoReader
torchvision.set_video_backend("video_reader")

reader = VideoReader("datasets/ffpp/real/c23/videos/000.avi", "video")
frame = next(reader)["data"]
print(frame.shape)
