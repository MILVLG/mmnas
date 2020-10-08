class Path:
    def __init__(self):
        self.DATASET_ROOT_PATH = './data/vgd/'
        # self.DATASET_ROOT_PATH = '/data/cuiyh/dataset/RefCOCO/'
        self.IMGFEAT_ROOT_PATH = './data/vgd/'
        # self.IMGFEAT_ROOT_PATH = '/data/cuiyh/features/'
        self.CKPT_PATH = './logs/ckpts/'

        self.IMGFEAT_PATH = {
            'vg_woref':{
                'train': self.IMGFEAT_ROOT_PATH + 'bua-r101-fix100/',
            },
            # 'coco_mrcn':{
            #     'refcoco': self.IMGFEAT_ROOT_PATH + 'vgd_coco/fix100/refcoco_unc/',
            #     'refcoco+': self.IMGFEAT_ROOT_PATH + 'vgd_coco/fix100/refcoco+_unc/',
            #     'refcocog': self.IMGFEAT_ROOT_PATH + 'vgd_coco/fix100/refcocog_umd/',
            # },
        }

        self.REF_PATH = {
            'refcoco': {
                'train': self.DATASET_ROOT_PATH + 'refcoco/train.json',
                'val': self.DATASET_ROOT_PATH + 'refcoco/val.json',
                'testA': self.DATASET_ROOT_PATH + 'refcoco/testA.json',
                'testB': self.DATASET_ROOT_PATH + 'refcoco/testB.json',
            },
            'refcoco+': {
                'train': self.DATASET_ROOT_PATH + 'refcoco+/train.json',
                'val': self.DATASET_ROOT_PATH + 'refcoco+/val.json',
                'testA': self.DATASET_ROOT_PATH + 'refcoco+/testA.json',
                'testB': self.DATASET_ROOT_PATH + 'refcoco+/testB.json',
            },
            'refcocog': {
                'train': self.DATASET_ROOT_PATH + 'refcocog/train.json',
                'val': self.DATASET_ROOT_PATH + 'refcocog/val.json',
                'test': self.DATASET_ROOT_PATH + 'refcocog/test.json',
            },
        }

        self.EVAL_PATH = {
            'result_test': self.CKPT_PATH + 'result_test/',
            'tmp': self.CKPT_PATH + 'tmp/',
            'arch': 'arch/',
        }