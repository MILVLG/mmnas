class Path:
    def __init__(self):
        self.DATASET_ROOT_PATH = './data/vqa/annotations'
        self.IMGFEAT_ROOT_PATH = './data/vqa/bua-r101-max100/'
        # self.IMGFEAT_ROOT_PATH = '/data-ssd/vqa/feats/'
        self.CKPT_PATH = './logs/ckpts/'

        self.IMGFEAT_PATH = {
            'train': self.IMGFEAT_ROOT_PATH + 'train2014/',
            'val': self.IMGFEAT_ROOT_PATH + 'val2014/',
            'test': self.IMGFEAT_ROOT_PATH + 'test2015/',
        }

        self.QUESTION_PATH = {
            'train': self.DATASET_ROOT_PATH + 'v2_OpenEnded_mscoco_train2014_questions.json',
            'train-anno': self.DATASET_ROOT_PATH + 'v2_mscoco_train2014_annotations.json',
            'val': self.DATASET_ROOT_PATH + 'v2_OpenEnded_mscoco_val2014_questions.json',
            'val-anno': self.DATASET_ROOT_PATH + 'v2_mscoco_val2014_annotations.json',
            'vg': self.DATASET_ROOT_PATH + 'VG_questions.json',
            'vg-anno': self.DATASET_ROOT_PATH + 'VG_annotations.json',
            'test': self.DATASET_ROOT_PATH + 'v2_OpenEnded_mscoco_test2015_questions.json',
        }

        self.EVAL_PATH = {
            'result_test': self.CKPT_PATH + 'result_test/',
            'tmp': self.CKPT_PATH + 'tmp/',
            'arch': 'arch/',
        }