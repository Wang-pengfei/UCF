# Uncertainty-aware Clustering for Unsupervised Domain Adaptive Object Re-identification (UCF)

The *official* repository for [Uncertainty-aware Clustering for Unsupervised Domain Adaptive Object Re-identification](https://arxiv.org/pdf/2108.09682.pdf).

### Updates

[2021-12-05] First submitted.

## Requirements

### Prepare Datasets

Download the person datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [DukeMTMC](https://arxiv.org/abs/1609.01775), [MSMT17](https://arxiv.org/abs/1711.08565), [PersonX](https://github.com/sxzrt/Instructions-of-the-PersonX-dataset#data-for-visda2020-chanllenge), and the vehicle datasets [VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html), [VeRi-776](https://github.com/JDAI-CV/VeRidataset), [VehicleX](https://www.aicitychallenge.org/2020-track2-download/).
Then unzip them under the directory like
```
data
├── market1501
│   └── Market-1501-v15.09.15
├── dukemtmc
│   └── DukeMTMC-reID
├── msmt17
│   └── MSMT17_V1
├── personx
│   └── PersonX
├── vehicleid
│   └── VehicleID_V1.0
├── vehiclex
│   └── AIC21_Track2_ReID_Simulation
└── veri
    └── VeRi
```

## Training

We utilize 4 GPUs for training. **Note that**

+ use `--width 128 --height 256` (default) for person datasets, and `--height 224 --width 224` for vehicle datasets;
+ use `-a resnet50` (default) for the backbone of ResNet-50.

### Unsupervised Domain Adaptation
To train the model(s) in the paper, run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/source_pretrained.py \
  -ds $SOURCE_DATASET -dt $TARGET_DATASET --logs-dir $PATH_OF_LOGS
```

**Some examples:**
```shell
### Market-1501 -> MSMT17 ###
# use all default settings is ok
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/source_pretrained.py \
  -ds market1501 -dt msmt17 --logs-dir logs/pretrained/market2msmt
# after pretraining , to train a baseline:
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/sbs_traindbscan.py \
  -ds market1501 -dt msmt17 --logs-dir logs/dbscan/market2msmt \
  --init-1 logs/pretrained/market2msmt/model_best.pth.tar
# after pretraining , to train a baseline + HC(Hierarchical clustering) + UCIS(uncertainty-aware collaborative instance selection) :
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/sbs_traindbscan.py \
  -ds market1501 -dt msmt17 --logs-dir logs/dbscan/market2msmt \
  --init-1 logs/pretrained/market2msmt/model_best.pth.tar \
  --HC --UCIS
```

## Evaluation

We utilize 4 GPUs for testing. **Note that**

+ use `--width 128 --height 256` (default) for person datasets, and `--height 224 --width 224` for vehicle datasets;
+ use `-a resnet50` (default) for the backbone of ResNet-50.

### Unsupervised Domain Adaptation

To evaluate the domain adaptive model on the **target-domain** dataset, run:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/sbs_traindbscan.py --evaluate \
  -dt $DATASET --init-1 $PATH_OF_MODEL
```

**Some examples:**
```shell
### Market-1501 -> MSMT17 ###
# test on the target domain
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/sbs_traindbscan.py --evaluate \
  -dt msmt17 --init-1 logs/dbscan/market2msmt/model_best.pth.tar
```

You can download the above models in the paper from [[Baidu Yun]](https://pan.baidu.com/s/1BHljorQHlGVuDeX0KhxgHw)(password: znoa).


## Citation
If you find this code useful for your research, please cite our paper
```
@article{wang2022uncertainty,
  title={Uncertainty-aware clustering for unsupervised domain adaptive object re-identification},
  author={Wang, Pengfei and Ding, Changxing and Tan, Wentao and Gong, Mingming and Jia, Kui and Tao, Dacheng},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}

```
