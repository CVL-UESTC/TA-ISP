# TA-ISP Object Detection (ROD dataset)

### Dataset Downloads
Download ROD dataset  from [百度网盘 (passwd: udqg)](https://pan.baidu.com/s/1aBroLAj_0XeyQSzJdKrJ7g?pwd=udqg). The dataset is randomly partitioned into a 9:1 train–validation split for both the day and night subsets using the following script:`./scripts/split_rod.py`

Unzip it under any folder, which format as:

```
-- ROD_TAISP
    -- annotations (detection label)
        --train_day640.json
        --train_night640.json
        --valid_day640.json
        --valid_night640.json
    -- day_raws_debayer_fp32_640x640
        -- Day
    -- night_raws_debayer_fp32_640x640
        -- Night
```
You can also download the original dataset (Only training dataset) from [DIAP](https://openi.pcl.ac.cn/innovation_contest/innov202305091731448/datasets?lang=en-US) and process the dataset using the script in `./scripts/process_raw.py` and `./scripts/process_anno.py`.


### Requirements
Codes are tested under python 3.9.19, torch 2.1.0, torchvision 0.16.0.

### Preprocessing
As mentioned in the paper, we use the pretrained weights of YOLOX-Tiny to initialize our model. So you need to download the pretrained weights from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth) and put it under `./pre-trained/`.

### Training
After all these preprocessing, run training simply by the following command. We also provide our hyper-parameters for training [here](./cfg_small.py).
```
bash train.sh
```

### Evaluation
To evaluate the model's performance, please follow these steps:

#### 1.Download Pretrained Weights:

Download our pretrained models from [releases](https://github.com/CVL-UESTC/TA-ISP/releases/tag/v0.0).

#### 2.Run Evaluation:

Execute the evaluation script by running the following command in your terminal:
```
bash eval.sh
```
