class Path:
    def __init__(self):
        self.DATASET_ROOT_PATH = './data/itm/'
        # self.DATASET_ROOT_PATH = '/data/cuiyh/dataset/retrival/'
        self.IMGFEAT_ROOT_PATH = './data/'
        # self.IMGFEAT_ROOT_PATH = '/data/cuiyh/features/'
        self.CKPT_PATH = './logs/ckpts/'

        self.IMGFEAT_PATH = {
            'coco': {
                'train': self.IMGFEAT_ROOT_PATH + 'rois_resnet101_fix36/train2014/',
                'val': self.IMGFEAT_ROOT_PATH + 'rois_resnet101_fix36/val2014/',
            },
            'flickr': {
                'train': self.IMGFEAT_ROOT_PATH + 'flickr_rois_resnet101_fix36/',
            },
        }
        # self.CAPTION_PATH = {
        #     'coco': {
        #         'train-caps': self.DATASET_ROOT_PATH + 'coco_precomp/train_caps.txt',
        #         'train-ids': self.DATASET_ROOT_PATH + 'coco_precomp/train_ids.txt',
        #         'dev-caps': self.DATASET_ROOT_PATH + 'coco_precomp/dev_caps.txt',
        #         'dev-ids': self.DATASET_ROOT_PATH + 'coco_precomp/dev_ids.txt',
        #         'test-caps': self.DATASET_ROOT_PATH + 'coco_precomp/test_caps.txt',
        #         'test-ids': self.DATASET_ROOT_PATH + 'coco_precomp/test_ids.txt',
        #         'testall-caps': self.DATASET_ROOT_PATH + 'coco_precomp/testall_caps.txt',
        #         'testall-ids': self.DATASET_ROOT_PATH + 'coco_precomp/testall_ids.txt',
        #     },
        #     'flickr': {
        #         'train-caps': self.DATASET_ROOT_PATH + 'f30k_precomp/train_caps.txt',
        #         'train-ids': self.DATASET_ROOT_PATH + 'f30k_precomp/train_ids.txt',
        #         'dev-caps': self.DATASET_ROOT_PATH + 'f30k_precomp/dev_caps.txt',
        #         'dev-ids': self.DATASET_ROOT_PATH + 'f30k_precomp/dev_ids.txt',
        #         'test-caps': self.DATASET_ROOT_PATH + 'f30k_precomp/test_caps.txt',
        #         'test-ids': self.DATASET_ROOT_PATH + 'f30k_precomp/test_ids.txt',
        #     },
        # }

        self.CAPTION_PATH = {
            'coco': {
                'train-caps': self.DATASET_ROOT_PATH + 'coco/train_caps.txt',
                'train-ids': self.DATASET_ROOT_PATH + 'coco/train_ids.txt',
                'dev-caps': self.DATASET_ROOT_PATH + 'coco/dev_caps.txt',
                'dev-ids': self.DATASET_ROOT_PATH + 'coco/dev_ids.txt',
                'test-caps': self.DATASET_ROOT_PATH + 'coco/test_caps.txt',
                'test-ids': self.DATASET_ROOT_PATH + 'coco/test_ids.txt',
                'testall-caps': self.DATASET_ROOT_PATH + 'coco/testall_caps.txt',
                'testall-ids': self.DATASET_ROOT_PATH + 'coco/testall_ids.txt',
            },
            'flickr': {
                'train-caps': self.DATASET_ROOT_PATH + 'flickr/train_caps.txt',
                'train-ids': self.DATASET_ROOT_PATH + 'flickr/train_ids.txt',
                'dev-caps': self.DATASET_ROOT_PATH + 'flickr/dev_caps.txt',
                'dev-ids': self.DATASET_ROOT_PATH + 'flickr/dev_ids.txt',
                'test-caps': self.DATASET_ROOT_PATH + 'flickr/test_caps.txt',
                'test-ids': self.DATASET_ROOT_PATH + 'flickr/test_ids.txt',
            },
        }

        self.EVAL_PATH = {
            'result_test': self.CKPT_PATH + 'result_test/',
            'tmp': self.CKPT_PATH + 'tmp/',
            'arch': self.CKPT_PATH + 'arch/',
        }