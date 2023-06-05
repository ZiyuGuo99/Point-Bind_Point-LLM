## Installation
Create a conda environment and install dependencies:
```bash
git clone https://github.com/ZrrSkywalker/Point-Bind.git
cd Point-Bind

conda create -n pointbind python=3.8
conda activate pointbind

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit

pip install -r requirements.txt
```
Install GPU-related packages:
```bash
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
Download pre-trained weights of Point-Bind with [I2P-MAE](https://drive.google.com/file/d/1V9y3h9EPlPN_HzU7zeeZ6xBOcvU-Xj6h/view?usp=sharing) or [PointBERT](https://drive.google.com/file/d/1BILH_aAGYuZOxvom8V9-n2fYW7nLGcai/view?usp=sharing) as 3D encoder, and organize the downloaded file in the following structure
  ```
  Point-Bind/
  ├── ckpts
      ├── pointbind_i2pmae.pt
      └── pointbind_pointbert.pt
  ```
