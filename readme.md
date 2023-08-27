# Installation Instructions
## Generic Packages
```shell
# Install torch with pip rather conda, due to: https://github.com/pytorch/pytorch/issues/102269
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# Install other dependencies
pip install lightning opencv-python-headless albumentations matplotlib
```
## Torchvision from source
```shell
git clone https://github.com/pytorch/vision.git
cd vision/
# version is versatile, select the latest stable version
git checkout tags/v0.15.2
conda install ffmpeg -c conda-forge
python setup.py install
```
