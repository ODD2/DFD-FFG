# Installation Instructions 
## Method 1
```shell
# Install torch with pip rather conda, due to: https://github.com/pytorch/pytorch/issues/102269
pip install torch torchaudio torchvision
# Install other dependencies
pip install lightning  lightning[pytorch-extra] 
pip install albumentations matplotlib ftfy regex opencv-python opencv-python-headless 
pip install pandas wandb open_clip_torch av
pip install face_alignment
```

## Method 2
```shell
# Install with conda environment file.
conda env create -f environment.yml
```

# Environment Variables
```shell
# The variables are set to send notification through Telegram. Please refer to the Telegram tutorial on how to create a chatbot and access the API_TOKEN and CHAT_ID.
API_TOKEN=''
CHAT_ID=''
```

# Dataset Structure
```shell
datasets
├── cdf
│   ├── FAKE
│   │   └── videos
│   │         └── *.avi
│   ├── REAL
│   │   └── videos
│   │         └── *.avi
│   └── csv_files
│       ├── test_fake.csv
│       └── test_real.csv
├── dfdc
│   ├── csv_files
│   │   └── test.csv
│   └── videos
├── dfo
│   ├── FAKE
│   │   └── videos
│   │         └── *.avi
│   ├── REAL
│   │   └── videos
│   │         └── *.avi
│   └── csv_files
│       ├── test_fake.csv
│       └── test_real.csv
├── ffpp
│   ├── DF
│   │   ├── c23
│   │   │   └── videos
│   │   │         └── *.avi
│   │   ├── c40
│   │   │   └── videos
│   │   │         └── *.avi
│   │   └── raw
│   │       └── videos
│   │   │         └── *.avi
│   ├── F2F ...
│   ├── FS ...
│   ├── FSh ...
│   ├── NT ...
│   ├── real ...
│   └── csv_files
│       ├── test.json
│       ├── train.json
│       └── val.json
|   
└── robustness
    ├── BW
    │   ├── 1
    │   │   ├── DF
    │   │   │   └── c23
    │   │   │       └── videos
    │   │   │             └── *.avi
    │   │   ├── F2F ...
    │   │   ├── FS ...
    │   │   ├── FSh ...
    │   │   ├── NT ...
    │   │   ├── real ...
    │   │   │
    │   │   └── csv_files
    │   │       ├── test.json
    │   │       ├── train.json
    │   │       └── val.json
    │   │   
    │   │   
    │   │   
    .   .
    .   .
    .   .
```


# General Preprocessing
```bash
# First, fetch all the landmarks & bboxes of the video frames.
python -m src.preprocess.fetch_landmark_bbox \ 
--root-dir="/storage/FaceForensicC23" \ # The root folder of the dataset
--video-dir="videos" \  # The root folder of the videos
--fdata-dir="frame_data" \ # The folder to save the extracted frame data
--glob-exp="*/*" \  # The glob expression to search through the root video folder
--split-num=1 \ # Split the dataset into several parts for parallel process.
--part-num=1 \ # The part of dataset to process  for parallel process.
--batch=1 \ # The batch size for the 2D-FAN face data extraction. (suggestion: 1)
--max-res=800  # The maximum resolution for either side of the image

# Then, crop all the faces from the original videos.
python -m src.preprocess.crop_main_face \ 
--root-dir="/storage/FaceForensicC23/" \ # The root folder of the dataset
--video-dir="videos" \  # The root folder of the videos
--fdata-dir="frame_data" \ # The folder to fetch the frame data for landmarks and bboxes
--glob-exp="*/*" \  # The glob expression to search through the root video folder
--crop-dir="cropped" \ # The folder to save the cropped videos
--crop-width=150 \ # The width for the cropped videos
--crop-height=150 \ # The height for the cropped videos
--mean-face="./misc/20words_mean_face.npy" # The mean face for face aligned cropping. 
--replace \ # Control whether to replace existing cropped videos
--workers \ # Number of works to perform parallel process (default: cpu / 2 )
```



# Robustness Preprocessing
```bash
# First, we add perturbation to all the videos.
python -m src.preprocess.phase1_apply_all_to_videos \ 
--dts-root="/storage/FaceForensicC23" \ # The root folder of the dataset
--vid-dir="videos" \  # The root folder of the videos
--rob-dir="robustness" \ # The folder to save the perturbed videos
--glob-exp="*/*.mp4" \  # The glob expression to search through the root video folder
--split=1 \ # Split the dataset into several parts for parallel process.
--part=1 \ # The part of dataset to process  for parallel process.
--workers=1 \ # Number of works to perform parallel process (default: cpu / 2 )

# Then, crop all the faces from the perturbed videos.
python -m src.preprocess.phase2_face_crop_all_videos \ 
(setup/run/clean) # the three phase operations
--root-dir="/storage/FaceForensicC23/" \ # The root folder of the dataset
--rob-dir="videos" \  # The root folder of the robustness videos
--fd-dir="frame_data" \ # The folder to fetch the frame data for landmarks and bboxes
--glob-exp="*/*/*/*.mp4" \  # The glob expression to search through the root video folder
--crop-dir="cropped_robust" \ # The folder to save the cropped videos
--mean-face="./misc/20words_mean_face.npy" # The mean face for face aligned cropping. 
--workers \ # Number of works to perform parallel process (default: cpu / 2 )
```

# Trainning


# Inference