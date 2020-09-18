# MmNas - Deep Multimodal Neural Architecture Search
This repository corresponds to the **PyTorch** implementation of the MmNas for {Visual Question Answering, Visual Grounding, Image-Text Matching}.

## Prerequisites

#### Software and Hardware Requirements

You may need a machine with at least **4 GPU (>= 8GB)**, **50GB memory for VQA and VGD** and **150GB for ITM** and **50GB free disk space**.  We strongly recommend to use a SSD drive to guarantee high-speed I/O.

You should first install some necessary packages.

1. Install [Python](https://www.python.org/downloads/) >= 3.6
2. Install [Cuda](https://developer.nvidia.com/cuda-toolkit) >= 9.0 and [cuDNN](https://developer.nvidia.com/cudnn)
3. Install [PyTorch](http://pytorch.org/) >= 0.4.1 with CUDA (**Pytorch 1.x is also supported**).
4. Install [SpaCy](https://spacy.io/) and initialize the [GloVe](https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz) as follows:

	```bash
	$ pip install -r requirements.txt
	$ wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
	$ pip install en_vectors_web_lg-2.1.0.tar.gz
	```


#### Setup for VQA

 The image features are extracted using the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) strategy, with each image being represented as an dynamic number (from 10 to 100) of 2048-D features. We store the features for each image in a `.npz` file. You can prepare the visual features by yourself or download the extracted features from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EsfBlbmK1QZFhCOFpr4c5HUBzUV0aH2h1McnPG1jWAxytQ?e=2BZl8O) or [BaiduYun](https://pan.baidu.com/s/1C7jIWgM3hFPv-YXJexItgw#list/path=%2F). The downloaded files contains three files: **train2014.tar.gz, val2014.tar.gz, and test2015.tar.gz**, corresponding to the features of the train/val/test images for *VQA-v2*, respectively. You should place them as follows:

```angular2html
|-- data
	|-- coco_extract
	|  |-- train2014.tar.gz
	|  |-- val2014.tar.gz
	|  |-- test2015.tar.gz
```

Besides, we use the VQA samples from the [visual genome dataset](http://visualgenome.org/) to expand the training samples. Similar to existing strategies, we preprocessed the samples by two rules:

1. Select the QA pairs with the corresponding images appear in the MSCOCO train and *val* splits.
2. Select the QA pairs with the answer appear in the processed answer list (occurs more than 8 times in whole *VQA-v2* answers).

For convenience, we provide our processed vg questions and annotations files, you can download them from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EmVHVeGdck1IifPczGmXoaMBFiSvsegA6tf_PqxL3HXclw) or [BaiduYun](https://pan.baidu.com/s/1QCOtSxJGQA01DnhUg7FFtQ#list/path=%2F), and place them as follow:


```angular2html
|-- datasets
	|-- vqa
	|  |-- VG_questions.json
	|  |-- VG_annotations.json
```

After that, you should:

1. Download the QA files for [VQA-v2](https://visualqa.org/download.html).
2. Unzip the bottom-up features

Finally, the `data` folders will have the following structure:

```angular2html
|-- data
	|-- coco_extract
	|  |-- train2014
	|  |  |-- COCO_train2014_...jpg.npz
	|  |  |-- ...
	|  |-- val2014
	|  |  |-- COCO_val2014_...jpg.npz
	|  |  |-- ...
	|  |-- test2015
	|  |  |-- COCO_test2015_...jpg.npz
	|  |  |-- ...
	|-- vqa
	|  |-- v2_OpenEnded_mscoco_train2014_questions.json
	|  |-- v2_OpenEnded_mscoco_val2014_questions.json
	|  |-- v2_OpenEnded_mscoco_test2015_questions.json
	|  |-- v2_OpenEnded_mscoco_test-dev2015_questions.json
	|  |-- v2_mscoco_train2014_annotations.json
	|  |-- v2_mscoco_val2014_annotations.json
	|  |-- VG_questions.json
	|  |-- VG_annotations.json

```

#### Setup for Visual Grounding

 The image features are extracted using the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) strategy, with two types featrues are used: 1. visual genome(W/O reference images) pre-trained faster-rcnn detector; 2. coco pre-trained mask-rcnn detector following [MAttNet](https://github.com/lichengunc/MAttNet). We store the features for each image in a `.npz` file. You can prepare the visual features by yourself or download the extracted features from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EsfBlbmK1QZFhCOFpr4c5HUBzUV0aH2h1McnPG1jWAxytQ?e=2BZl8O) and place in ./data folder.

 Refs dataset{refcoco, refcoco+, recocog} were introduced [here](https://github.com/lichengunc/refer), build and place them as follow:


```angular2html
|-- data
	|-- vgd_coco
	|  |-- fix100
	|  |  |-- refcoco_unc
	|  |  |-- refcoco+_unc
	|  |  |-- refcocg_umd
	|-- detfeat100_woref
	|-- refs
	|  |-- refcoco
	|  |   |-- instances.json
	|  |   |-- refs(google).p
	|  |   |-- refs(unc).p
	|  |-- refcoco+
	|  |   |-- instances.json
	|  |   |-- refs(unc).p
	|  |-- refcocog
	|  |   |-- instances.json
	|  |   |-- refs(google).p
```

Additionally, it is also needed to bulid as follows:
```
cd mmnas/utils
python3 setup.py build
cp build/[lib.*/*.so] .
cd ../..
```


#### Setup for Image-Text Matching

 The image features are extracted using the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) strategy, with each image being represented as an dynamic number (fixed 36) of 2048-D features. We store the features for each image in a `.npz` file. You can prepare the visual features by yourself or download the extracted features from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EsfBlbmK1QZFhCOFpr4c5HUBzUV0aH2h1McnPG1jWAxytQ?e=2BZl8O) and place in ./data folder.

 Retrival dataset{flickr, coco} can be found [here](https://scanproject.blob.core.windows.net/scan-data/data_no_feature.zip), extract and place them as follow:


```angular2html
|-- data
	|-- rois_resnet101_fix36
	|  |-- train2014
	|  |-- val2014
	|-- flickr_rois_resnet101_fix36
	|-- itm
	|  |-- coco_precomp
	|  |-- f30k_precomp
```


## Training

The following script will start training with the default hyperparameters:

1. VQA

```bash
$ python3 train_vqa.py --RUN='train' --GENO_PATH='./logs/ckpts/arch/train_vqa.json'
```

2. VGD

```bash
$ python3 train_vgd.py --RUN='train' --GENO_PATH='./logs/ckpts/arch/train_vgd.json'
```

3. ITM

```bash
$ python3 train_itm.py --RUN='train' --GENO_PATH='./logs/ckpts/arch/train_itm.json'
```

To add：

1. ```--VERSION=str```, e.g.```--VERSION='small_model'``` to assign a name for your this model.

2. ```--GPU=str```, e.g.```--GPU='0, 1, 2, 3'``` to train the model on specified GPU device.

3. ```--NW=int```, e.g.```--NW=8``` to accelerate I/O speed.

4. ```--MODEL={'small', 'large'}```  ( Warning: The large model will consume more GPU memory, maybe [Multi-GPU Training and Gradient Accumulation](#Multi-GPU-Training-and-Gradient-Accumulation) can help if you want to train the model with limited GPU memory.)

5. ```--SPLIT={'train', 'train+val', 'train+val+vg'}``` can combine the training datasets as you want. The default training split is ```'train+val+vg'```.  Setting ```--SPLIT='train'```  will trigger the evaluation script to run the validation score after every epoch automatically.

6. ```--RESUME``` to start training with saved checkpoint parameters.

6. ```--GENO_PATH``` can use the different searched architectures.



## Validation and Testing

**Warning**: If you train the model use ```--MODEL``` args or multi-gpu training, it should be also set in evaluation.


#### Offline Evaluation

It is a easy way to modify follows args: --RUN={'val', 'test'} --CKPT_PATH=[Your Model Path] to **Run val or test Split**.

Example:

```bash
$ python3 train_vqa.py --RUN='test' --CKPT_PATH=[Your Model Path] --GENO_PATH=[Searched Architecture Path]
```


#### Online Evaluation (ONLY FOR VQA)

Test Result files will stored in ```./logs/ckpts/result_test/result_run_[Your Version].json```

You can upload the obtained result json file to [Eval AI](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) to evaluate the scores on *test-dev* and *test-std* splits.


## Citation

If this repository is helpful for your research, we'd really appreciate it if you could cite the following paper:

```
@article{yu2020deep,
  title={Deep Multimodal Neural Architecture Search},
  author={Yu, Zhou and Cui, Yuhao and Yu, Jun and Wang, Meng and Tao, Dacheng and Tian, Qi},
  journal={arXiv preprint arXiv:2004.12070},
  year={2020}
}
```


