# PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment

Modifications to the original repository for PANet

### Dependencies

* Python 3.6 +
* PyTorch 1.0.1
* torchvision 0.2.1
* NumPy, SciPy, PIL
* pycocotools
* sacred 0.7.5
* tqdm 4.32.2

### Data Preparation

The prepared data containing both VOC and COCO datasets can be found [here] (https://drive.google.com/file/d/1OGY-yDE01GETtNbJGhx94mKIqT0zwxPR/view?usp=sharing). If you want to setup the data manually, please follow the instructions below

#### VOC Dataset

1. Download `SegmentationClassAug`, `SegmentationObjectAug`, `ScribbleAugAuto` from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp?usp=sharing) and put them under `VOCdevkit/VOC2012`.
2. Download `Segmentation` from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp?usp=sharing) and use it to replace `VOCdevkit/VOC2012/ImageSets/Segmentation`.


### Usage

1. Download the ImageNet-pretrained weights of VGG16 network from `torchvision`: [https://download.pytorch.org/models/vgg16-397923af.pth](https://download.pytorch.org/models/vgg16-397923af.pth) and put it under `PANet/pretrained_model` folder.

2. Change configuration via `config.py`, then train the model using `python train.py` or test the model using `python test.py`. You can use `sacred` features, e.g. `python train.py with gpu_id=2`.
