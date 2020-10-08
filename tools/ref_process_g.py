import json, pickle
import numpy as np

splitby = 'umd'
dataset = 'refcocog'

instances = json.load(open('../data/vgd/' + dataset + '/instances.json', 'r'))
refs = pickle.load(open('../data/vgd/' + dataset +
                        '/refs(' + splitby + ').p', 'rb'))
sample_file_train = open('../data/vgd/' + dataset + '/train.json', 'w')
sample_file_val = open('../data/vgd/' + dataset + '/val.json', 'w')
sample_file_test = open('../data/vgd/' + dataset + '/test.json', 'w')

train_sample = []
val_sample = []
test_sample = []

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
        elif ref['split'] == 'test':
            test_sample.append(sub_sample)
        else:
            pass

print(train_sample.__len__())
print(val_sample.__len__())
print(test_sample.__len__())
#
json.dump(train_sample, sample_file_train)
json.dump(val_sample, sample_file_val)
json.dump(test_sample, sample_file_test)

