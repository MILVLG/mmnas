import numpy as np
import glob, json, re, torch, en_vectors_web_lg
import torch.utils.data as Data
from mmnas.utils.answer_punct import preprocess_answer


def relation_embedding(f_g):
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)  # [n_obj, 1]

    cx = (x_min + x_max) * 0.5  # [n_obj, 1]
    cy = (y_min + y_max) * 0.5  # [n_obj, 1]
    w = (x_max - x_min) + 1.  # [n_obj, 1]
    h = (y_max - y_min) + 1.  # [n_obj, 1]

    delta_x = cx - cx.view(1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)  # [n_obj, n_obj]

    delta_y = cy - cy.view(1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)  # [n_obj, n_obj]

    delta_w = torch.log(w / w.view(1, -1))  # [n_obj, n_obj]
    delta_h = torch.log(h / h.view(1, -1))  # [n_obj, n_obj]
    size = delta_h.size()

    delta_x = delta_x.view(size[0], size[1], 1)
    delta_y = delta_y.view(size[0], size[1], 1)
    delta_w = delta_w.view(size[0], size[1], 1)
    delta_h = delta_h.view(size[0], size[1], 1)  # [n_obj, n_obj, 1]
    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)  # [n_obj, n_obj, 4]

    return position_mat


def semantic_embedding(ques, ques_ix, pretrained_emb, max_token=14):
    words = re.sub(r"([.,'!?\"()*#:;])", '', ques['question'].lower()).replace('-', ' ').replace('/', ' ').split()
    size = min(len(words), max_token)

    # proc glove
    words_glove = torch.zeros((size, 300)).float()
    for ix, word_ix in enumerate(ques_ix[:size]):
        words_glove[ix] = torch.from_numpy(pretrained_emb[word_ix])

    sub_glove = words_glove.view(size, 1, 300) - words_glove.view(1, size, 300)
    glove_l2 = torch.norm(sub_glove, dim=-1)

    mul_glove = words_glove.view(size, 1, 300) * words_glove.view(1, size, 300)
    mod_glove = torch.sqrt(torch.norm(words_glove, dim=-1))
    glove_cos = torch.sum(mul_glove, dim=-1) / (mod_glove.view(size, 1) * mod_glove.view(1, size) + 1e-6)

    # proc position
    words_position = torch.arange(size).float()
    sub_position = torch.abs(words_position.view(-1, 1) - words_position.view(1, -1)) / size

    cat_emb = torch.cat((glove_l2.view(size, size, 1), glove_cos.view(size, size, 1), sub_position.view(size, size, 1)), dim=-1)

    return cat_emb


class DataSet(Data.Dataset):
    def __init__(self, __C, RUN_MODE):
        self.__C = __C
        self.RUN_MODE = RUN_MODE

        # Loading all image paths
        frcn_feat_path_list = \
            glob.glob(__C.IMGFEAT_PATH['train'] + '*.npz') + \
            glob.glob(__C.IMGFEAT_PATH['val'] + '*.npz') + \
            glob.glob(__C.IMGFEAT_PATH['test'] + '*.npz')

        # Loading question word list
        stat_ques_list = \
            json.load(open(__C.QUESTION_PATH['train'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['val'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['test'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['vg'], 'r'))['questions']

        # Loading answer word list
        stat_ans_list = \
            json.load(open(__C.QUESTION_PATH['train-anno'], 'r'))['annotations'] + \
            json.load(open(__C.QUESTION_PATH['val-anno'], 'r'))['annotations']

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        split_list = __C.SPLIT[RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.QUESTION_PATH[split], 'r'))['questions']
            if RUN_MODE in ['train']:
                self.ans_list += json.load(open(__C.QUESTION_PATH[split + '-anno'], 'r'))['annotations']

        # Define run data size
        if RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print(' ========== Dataset size:', self.data_size)


        # {image id} -> {image feature absolutely path}
        self.iid_to_frcn_feat_path = self.img_feat_path_load(frcn_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = self.ques_load(self.ques_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = self.tokenize(stat_ques_list, __C.GLOVE_FEATURE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)

        # Answers statistic
        self.ans_to_ix, self.ix_to_ans = self.ans_stat(stat_ans_list, ans_freq=8)
        self.ans_size = self.ans_to_ix.__len__()
        print(' ========== Answer token vocab size (occur more than {} times):'.format(8), self.ans_size)



    def img_feat_path_load(self, path_list):
        iid_to_path = {}

        for ix, path in enumerate(path_list):
            iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
            # print(iid)
            iid_to_path[iid] = path

        return iid_to_path


    def ques_load(self, ques_list):
        qid_to_ques = {}

        for ques in ques_list:
            qid = str(ques['question_id'])
            qid_to_ques[qid] = ques

        return qid_to_ques


    def tokenize(self, stat_ques_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        for ques in stat_ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques['question'].lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb


    def ans_stat(self, stat_ans_list, ans_freq):
        ans_to_ix = {}
        ix_to_ans = {}
        ans_freq_dict = {}

        for ans in stat_ans_list:
            ans_proc = preprocess_answer(ans['multiple_choice_answer'])
            if ans_proc not in ans_freq_dict:
                ans_freq_dict[ans_proc] = 1
            else:
                ans_freq_dict[ans_proc] += 1

        ans_freq_filter = ans_freq_dict.copy()
        for ans in ans_freq_dict:
            if ans_freq_dict[ans] <= ans_freq:
                ans_freq_filter.pop(ans)

        for ans in ans_freq_filter:
            ix_to_ans[ans_to_ix.__len__()] = ans
            ans_to_ix[ans] = ans_to_ix.__len__()

        return ans_to_ix, ix_to_ans



    def __getitem__(self, idx):
        frcn_feat_iter = np.zeros(1)
        bbox_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)

        if self.RUN_MODE in ['train']:
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]
            iid = str(ans['image_id'])

            ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)
            ans_iter = self.proc_ans(ans, self.ans_to_ix)

        else:
            ques = self.ques_list[idx]
            iid = str(ques['image_id'])

            ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)

        rel_ques = semantic_embedding(ques, ques_ix_iter, self.pretrained_emb, max_token=14)
        rel_ques_iter = torch.zeros(14, 14, 3)
        rel_ques_iter[:rel_ques.size(0), :rel_ques.size(1), :] = rel_ques[:]

        frcn_feat = np.load(self.iid_to_frcn_feat_path[iid])
        frcn_feat_x = frcn_feat['x'].transpose((1, 0))
        frcn_feat_iter = self.proc_img_feat(frcn_feat_x, img_feat_pad_size=100)

        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['image_h'], frcn_feat['image_w'])
            ),
            img_feat_pad_size=100
        )

        bbox = torch.from_numpy(frcn_feat['bbox'].astype(np.float32))
        rel_img = relation_embedding(bbox)  # [n_obj, n_obj, 4]
        rel_img_iter = torch.zeros(100, 100, 4)
        rel_img_iter[:rel_img.size(0), :rel_img.size(1), :] = rel_img[:]

        return torch.from_numpy(frcn_feat_iter), \
               torch.from_numpy(bbox_feat_iter), \
               rel_img_iter, \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter), \
               rel_ques_iter

    def __len__(self):
        return self.data_size


    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat


    def proc_bbox_feat(self, bbox, img_shape):
        bbox_feat = np.zeros((bbox.shape[0], 5), dtype=np.float32)

        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox_feat


    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix


    def get_score(self, occur):
        if occur == 0:
            return .0
        elif occur == 1:
            return .3
        elif occur == 2:
            return .6
        elif occur == 3:
            return .9
        else:
            return 1.


    def proc_ans(self, ans, ans_to_ix):
        ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
        ans_prob_dict = {}

        for ans_ in ans['answers']:
            ans_proc = preprocess_answer(ans_['answer'])
            if ans_proc not in ans_prob_dict:
                ans_prob_dict[ans_proc] = 1
            else:
                ans_prob_dict[ans_proc] += 1

        # if self.__C.LOSS_FUNC in ['kld']:
        #     for ans_ in ans_prob_dict:
        #         if ans_ in ans_to_ix:
        #             ans_score[ans_to_ix[ans_]] = ans_prob_dict[ans_] / 10.
        # else:
        for ans_ in ans_prob_dict:
            if ans_ in ans_to_ix:
                ans_score[ans_to_ix[ans_]] = self.get_score(ans_prob_dict[ans_])

        return ans_score
