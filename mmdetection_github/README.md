# TA-ISP Object Detection (LOD dataset, PASCAL RAW dataset)

### 📖 1: Dataset Download

**LOD Dataset** (low-light RAW detection dataset):

Download LOD dataset  from [Google Drive](https://drive.google.com/file/d/1Jkm4mvynWxc7lXSH3H9sLI0wJ6p6ftvZ/view?usp=sharing) or [百度网盘 (passwd: kf43)](https://pan.baidu.com/s/1FA9lw1WXk2dJ0jtlLeho5w), or find the LOD dataset [original provide link](https://github.com/ying-fu/LODDataset).

Unzip and place it in $./data$ under this folder, which format as:

```
--  data
     -- LOD_BMVC21
         -- RAW_dark (RAW data, "demosacing" in our paper)
         -- RGB_dark (default ISP RGB data)
         -- RAW_dark_InverseISP (InvISP processed RAW data, [CVPR 2021])
         -- RAW_dark_ECCV16_Micheal (ECCV16 ISP processed RAW data, [ECCV 2016])
         -- RAW-dark-Annotations (detection label)      
         -- trainval
```

**PASCAL RAW Dataset** (RAW detection dataset):

Download LOD dataset from [Google Drive](https://drive.google.com/file/d/1686W89ALVvtfUvK8NMvqWaUCTLBqhW-p/view?usp=sharing) or [百度网盘 (passwd: kjv9)](https://pan.baidu.com/s/1O76R8ZFZdLw88N0b3hT2Tw).

Unzip and place it in $./data$ under this folder, which format as:

```
--  data
     -- PASCAL_RAW
         -- annotations
         -- original (original RAW, demosaic RAW normal-light & over-exposure & low-light)
         -- compare_ISP (ISP methods, InvISP, ECCV16-ISP)
         -- trainval
```

**Note:** the original RAW file in PASCAL RAW are too big (>100GB), you could download them from [PASCAL RAW webiste](https://purl.stanford.edu/hq050zr7488), the code translate original RAW data to demosaic RAW data (normal-light, over-exposure, low-light) could find in here: [PASCAL_RAW_pre_process.py](PASCAL_RAW_pre_process.py).


### 📖 2: Enviroment Setup

Our code based on [mmdetection 3.3.0](https://github.com/open-mmlab/mmdetection?tab=readme-ov-file) version, you can following their [instruction](https://mmdetection.readthedocs.io/en/latest/get_started.html) to build environment. Or following our steps below:

(1). Create a conda environment and activate it:
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

(2). Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g. ours (torch1.12.1+cu113):
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

(3). Mmdetection setup:
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

Develop mmdet and install rawpy:
```
pip install -v -e .
pip install rawpy
```

### 📖 3: Model Evaluation

Check the whole pretrain weights at [release](https://github.com/CVL-UESTC/TA-ISP/releases).

#### 3.1: **LOD Dataset** config & pretrain weights (ckpt).

RetinaNet - ResNet50 backbone: 

| **TA-ISP** | 63.9 | [config](configs/LOD/R_Net_taisp.py) | [[ckpt]](https://github.com/CVL-UESTC/TA-ISP/releases/download/v0.0/lod_RNet_res50_taisp.pth)


Evaluation of TA-ISP, only need single GPU (TA-ISP, RetinaNet for example), if you need visulization, please add "--show-dir": 

```
python tools/test.py configs/LOD/R_Net_taisp.py https://github.com/CVL-UESTC/TA-ISP/releases/download/v0.0/lod_RNet_res50_taisp.pth
```

#### 3.2: **PASCAL RAW Dataset** config & pretrain weights (ckpt):

RetinaNet - ResNet18 backbone: 

|  **TA-ISP** | 89.9 | [config](configs/PASCALRAW_Res18/Normal_Light_taisp_res18.py) | [[ckpt]](https://github.com/CVL-UESTC/TA-ISP/releases/download/v0.0/pascalraw_res18_taisp.pth)

RetinaNet - ResNet50 backbone: 

| **TA-ISP** | 90.2 | [config](configs/PASCALRAW_Res50/Normal_Light_raw_adapter_res50.py) | [[ckpt]](https://github.com/CVL-UESTC/TA-ISP/releases/download/v0.0/pascalraw_res50_taisp.pth)

Evaluation of TA-ISP, only need single GPU (TA-ISP, ResNet18), if you need visulization, please add "--show-dir": 

```
python tools/test.py configs/PASCALRAW_Res18/Normal_Light_taisp_res18.py https://github.com/CVL-UESTC/TA-ISP/releases/download/v0.0/pascalraw_res18_taisp.pth
```

### 📖 4: Model Training (Optional)

```
sh tools/train.sh
```


### 📖 Acknowledgement:

We thanks mmdetection & LOD & PASCAL RAW for their excellent code base & dataset.
