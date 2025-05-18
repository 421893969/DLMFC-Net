# Dual-Level Modality Fusion Compensation: Enhancing Visible-Infrared Person Re-identification via Image and Feature Level Integration 
 Datasets: Dataset RegDB [1] dan dataset SYSU-MM01 [2].

|Datasets    | Pretrained| Rank@1  | mAP |  mINP |  Model|
| --------   | -----    | -----  |  -----  | ----- |------|
|#RegDB      | ImageNet | ~ 95.84% | ~ 90.78%| -----|----- |
|#SYSU-MM01  | ImageNet | ~ 66.74%  | ~ 63.72% | -----|


### 1. Prepare the datasets.

- (1) RegDB Dataset [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

- (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 


- (2) SYSU-MM01 Dataset [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

- run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

### 2. Joint Training.
Train a model by
```bash
python train.py --dataset sysu --lr 0.1 --method adp --augc 1 --rande 0.5 --gpu 1
```

- `--dataset`: which dataset "sysu" or "regdb".

- `--lr`: initial learning rate.

-  `--method`: method to run Enhanced Squared Difference or Baseline.

-  `--augc`:  Channel augmentation or not.

-  `--rande`:  random erasing with probability.

- `--gpu`:  which gpu to run.

You may need mannully define the data path first.

**Parameters**: More parameters can be found in the script.

**Sampling Strategy**: N (= bacth size) person identities are randomly sampled at each step, then randomly select four visible and four thermal image. Details can be found in Line 302-307 in `train.py`.

**Training Log**: The training log will be saved in `log/" dataset_name"+ log`. Model will be saved in `save_model/`.

### 3. Testing.

Test a model on SYSU-MM01 or RegDB dataset 
```bash
python testa.py --mode all --resume 'model_path' --gpu 1 --dataset sysu
```
- `--dataset`: which dataset "sysu" or "regdb".

- `--mode`: "all" or "indoor" all search or indoor search (only for sysu dataset).

- `--trial`: testing trial (only for RegDB dataset).

- `--resume`: the saved model path.

- `--gpu`:  which gpu to run.

### 4. Citation

Please kindly cite this paper in your publications if it helps your research:
```
@inproceedings{iccv21caj,
author    = {Ye, Mang and Ruan, Weijian and Du, Bo and Shou, Mike Zheng},
title     = {Channel Augmented Joint Learning for Visible-Infrared Recognition},
booktitle = {IEEE/CVF International Conference on Computer Vision},
year      = {2021},
pages     = {13567-13576}
}
```

###  5. References.

[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.




Contact: jiangtaoguoseu.edu.cn
