import torch
import os
import sys
os.chdir(os.environ["PROJECT_ROOT"])
sys.path.append(os.environ["PROJECT_ROOT"])


@torch.inference_mode()
def extract_features(encoder, frame):
    assert len(frame.shape) == 4
    # get attributes from each CLIP ViT layer
    kvs = encoder(frame.to("cuda"))
    # discard original CLS token and restore temporal dimension
    for i in range(len(kvs)):
        for k in kvs[i].keys():
            kvs[i][k] = kvs[i][k][:, 1:].to("cpu")
            if (not k == "out"):
                kvs[i][k] = kvs[i][k].flatten(-2)
    torch.cuda.empty_cache()
    return kvs
