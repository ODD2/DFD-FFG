# %%
import cv2
from glob import glob
from tqdm import tqdm

VIDEO_DIR = "/scratch1/users/od/CelebDF/Real/videos/*.mp4"
ROB_DIR = "robustness"

videos = glob(VIDEO_DIR)
type_list = ['CS', 'CC', 'BW', 'GNC', 'GB', 'JPEG', 'VC']
level_list = [1, 2, 3, 4, 5]

unmatch = []
for video in tqdm(videos):
    cap = cv2.VideoCapture(video)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    for t in type_list:
        for l in level_list:
            rob_video = video.replace("videos", f"robustness/{t}/{l}")
            cap = cv2.VideoCapture(rob_video)
            if (not frames == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                unmatch.append(rob_video)
            cap.release()
print(unmatch)
# %%
for i in unmatch:
    print(i)
# %%
