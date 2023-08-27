from os import path, makedirs
import matplotlib.pyplot as plt


def dataset_entity_visualize(entity_data, normalized=False, unit=2, base_dir="./misc/extern/test/"):
    makedirs(base_dir, exist_ok=True)
    clips = entity_data['clips']
    comp = entity_data['comp']
    idx = entity_data['idx']
    df_type = entity_data['df_type']
    vid_path = entity_data['vid_path']
    num_clips, num_frames = clips.shape[:2]
    plt.figure(figsize=(unit*num_frames*0.9, unit*num_clips), layout="constrained")
    plt.suptitle(f"#{idx},{df_type},{comp}\n{vid_path}", fontsize=unit*9)
    for i, clip in enumerate(clips):
        plt.subplot(num_clips, 1, i+1)
        plt.gca().axis('off')
        plt.imshow(clip.permute(2, 0, 3, 1).flatten(1, 2).numpy())
    plt.savefig(path.join(base_dir, f"{idx}.jpg"))
    plt.close()
