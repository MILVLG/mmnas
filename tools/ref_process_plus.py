import json, pickle
import numpy as np

splitby = 'unc'
dataset = 'refcoco+'

instances = json.load(open('../data/vgd/' + dataset + '/instances.json', 'r'))
refs = pickle.load(open('../data/vgd/' + dataset +
                        '/refs(' + splitby + ').p', 'rb'))
sample_file_train = open('../data/vgd/' + dataset + '/train.json', 'w')
sample_file_val = open('../data/vgd/' + dataset + '/val.json', 'w')
sample_file_testA = open('../data/vgd/' + dataset + '/testA.json', 'w')
sample_file_testB = open('../data/vgd/' + dataset + '/testB.json', 'w')

train_sample = []
val_sample = []
testA_sample = []
testB_sample = []

imgid2img = {}
annid2ann = {}
catid2ann = {}
for img in instances['images']:
    if img['id'] not in imgid2img:
        imgid2img[img['id']] = img
for ann in instances['annotations']:
    if ann['id'] not in annid2ann:
        annid2ann[ann['id']] = ann
for cat in instances['categories']:
    if cat['id'] not in catid2ann:
        catid2ann[cat['id']] = cat

for ref in refs:
    for sent in ref['sentences']:
        sub_sample = {}
        sub_sample['tokens'] = sent['tokens']
        sub_sample['file_name'] = ref['file_name']
        sub_sample['image_id'] = ref['image_id']
        sub_sample['split'] = ref['split']
        sub_sample['bbox'] = annid2ann[ref['ann_id']]['bbox']
        sub_sample['name'] = catid2ann[ref['category_id']]['name']
        sub_sample['height'] = imgid2img[ref['image_id']]['height']
        sub_sample['width'] = imgid2img[ref['image_id']]['width']


        if ref['split'] == 'train':
            train_sample.append(sub_sample)
        elif ref['split'] == 'val':
            val_sample.append(sub_sample)
        elif ref['split'] == 'testA':
            testA_sample.append(sub_sample)
        elif ref['split'] == 'testB':
            testB_sample.append(sub_sample)
        else:
            pass

print(train_sample.__len__())
print(val_sample.__len__())
print(testA_sample.__len__())
print(testB_sample.__len__())
'''
refcoco: {'train': 120624, 'val': 10834, 'testA': 5657, 'testB': 5095}
refcoco+: {'train': 120191, 'val': 10758, 'testA': 5726, 'testB': 4889}
{'tokens': ['the', 'lady', 'with', 'the', 'blue', 'shirt'], 'file_name': 'COCO_train2014_000000581857_16.jpg', 'image_id': 581857, 'split': 'train', 'bbox': [103.93, 299.99, 134.22, 177.42], 'name': 'person', 'height': 640, 'width': 427}
'''
json.dump(train_sample, sample_file_train)
json.dump(val_sample, sample_file_val)
json.dump(testA_sample, sample_file_testA)
json.dump(testB_sample, sample_file_testB)

