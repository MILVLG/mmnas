# Dataset Setup


## VQA Setup

The image features are extracted using the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) strategy, with each image being represented as a dynamic number (from 10 to 100) of 2048-D features. We store the features for each image in a `.npz` file. You can prepare the visual features by yourself or download the extracted features from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EsfBlbmK1QZFhCOFpr4c5HUBzUV0aH2h1McnPG1jWAxytQ?e=2BZl8O) or [BaiduYun](https://pan.baidu.com/s/1C7jIWgM3hFPv-YXJexItgw#list/path=%2F). The downloaded files contains three files: **train2014.tar.gz, val2014.tar.gz, and test2015.tar.gz**, corresponding to the features of the train/val/test images for *VQA-v2*, respectively. Run the following commands to unzip the features:
<!-- 
```angular2html
|-- data
	|-- bua-r101-max100
	|  |-- train2014.tar.gz
	|  |-- val2014.tar.gz
	|  |-- test2015.tar.gz
``` -->
```Bash
mkdir data/vqa/bua-r101-max100
tar -xzvf train2014.tar.gz -C data/vqa/bua-r101-max100/
tar -xzvf val2014.tar.gz -C data/vqa/bua-r101-max100/
tar -xzvf test2015.tar.gz -C data/vqa/bua-r101-max100/
```

Then download the QA files for [VQA-v2](https://visualqa.org/download.html). Besides, we use the VQA samples from the [visual genome dataset](http://visualgenome.org/) to expand the training samples. Similar to existing strategies, we preprocessed the samples by two rules:

1. Select the QA pairs with the corresponding images appear in the MSCOCO train and *val* splits.
2. Select the QA pairs with the answer appear in the processed answer list (occurs more than 8 times in whole *VQA-v2* answers).

For convenience, we provide our processed vg questions and annotations files, you can download them from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EmVHVeGdck1IifPczGmXoaMBFiSvsegA6tf_PqxL3HXclw) or [BaiduYun](https://pan.baidu.com/s/1QCOtSxJGQA01DnhUg7FFtQ#list/path=%2F). Place all annotation files into the folder `data/vqa/annotations`.

Finally, the `data` folder will have the following structure:

```angular2html
|-- data
	|-- vqa
	    |-- bua-r101-max100
	    |   |-- train2014
	    |   |   |-- COCO_train2014_...jpg.npz
	    |   |   |-- ...
	    |   |-- val2014
	    |   |   |-- COCO_val2014_...jpg.npz
	    |   |   |-- ...
	    |   |-- test2015
	    |   |   |-- COCO_test2015_...jpg.npz
	    |   |   |-- ...
		|-- annotations
	    |   |-- v2_OpenEnded_mscoco_train2014_questions.json
	    |   |-- v2_OpenEnded_mscoco_val2014_questions.json
	    |   |-- v2_OpenEnded_mscoco_test2015_questions.json
	    |   |-- v2_OpenEnded_mscoco_test-dev2015_questions.json
	    |   |-- v2_mscoco_train2014_annotations.json
	    |   |-- v2_mscoco_val2014_annotations.json
	    |   |-- VG_questions.json
	    |   |-- VG_annotations.json

```

## VGD Setup

The image features are extracted using the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) strategy, with each image being represented as a fixed number (fixed 100) of 2048-D features. A visual genome(W/O reference images) pre-trained faster-rcnn detector is used to extract features. We store the features for each image in a `.npz` file. You can prepare the visual features by yourself or download the extracted features from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/Ehz5A3Eif-JHhZTLhxs7vrEBrDbCEKBUDto4J57fA0GCDg?e=jkTLUy) and run the following commands:

```Bash
cat vgd-bua-fix100.tar.gz* | tar xz
mv vgd-bua-fix100 data/vgd/bua-r101-fix100
```

Refs dataset{refcoco, refcoco+, recocog} can be downloaded from [here](https://github.com/lichengunc/refer), then conduct the preprocessing procedures as follows:

```Bash
python tools/ref_process.py
python tools/ref_process_plus.py
python tools/ref_process_g.py
```

Finally, the `data` folder will have the following structure:

```angular2html
|-- data
	|-- vgd
	    |-- bua-r101-fix100
	    |-- refcoco
	    |   |-- train.json
	    |   |-- val.json
	    |   |-- testA.json
	    |   |-- testB.json
	    |-- refcoco+
	    |   |-- train.json
	    |   |-- val.json
	    |   |-- testA.json
	    |   |-- testB.json
	    |-- refcocog
	    |   |-- train.json
	    |   |-- val.json
	    |   |-- test.json
```

Additionally, it is also needed to build as follows:
```
cd mmnas/utils
python3 setup.py build
cp build/lib.*/*.so .
cd ../..
```


## ITM Setup

The image features are extracted using the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) strategy, with each image being represented as a fixed number (fixed 36) of 2048-D features. We store the features for each image in a `.npz` file. You can prepare the visual features by yourself or download the extracted features from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EtbW4UUOn81CgRbIhRzsJUwBEEZDCGQU1oiuhhcBUbEC9Q?e=iCwNdi) and run the following commands:

```Bash
cat itm-bua-fix36.tar.gz* | tar xz
mv vgd-bua-fix100 data/vgd/bua-r101-fix36
```

Retrival dataset can be found from [here](https://scanproject.blob.core.windows.net/scan-data/data_no_feature.zip). Extract the `f30k_precomp` folder and place it into `./data/itm/`.

Finally, the `./data` folder will have the following structure:
```angular2html
|-- data
	|-- itm
	    |-- flickr_bua-r101-fix36
	    |-- dataset.json
	    |-- f30k_precomp
```
