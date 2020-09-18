import numpy as np
import glob, json, re, torch, en_vectors_web_lg, random
import torch.utils.data as Data

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


class DataSet(Data.Dataset):
    def __init__(self, __C, RUN_MODE):
        self.__C = __C
        self.RUN_MODE = RUN_MODE

        # Loading all image paths
        frcn_feat_path_list = []
        for feat_split in __C.IMGFEAT_PATH[__C.DATASET]:
            frcn_feat_path_list += glob.glob(__C.IMGFEAT_PATH[__C.DATASET][feat_split] + '*.npz')

        # Loading question word list
        stat_caps_list = []
        for cap_split in __C.CAPTION_PATH[__C.DATASET]:
            if 'caps' in cap_split:
                with open(__C.CAPTION_PATH[__C.DATASET][cap_split], "r") as f:
                    for line in f:
                        # print(line)
                        stat_caps_list.append(line.strip())


        # Loading question and answer list
        self.caps_list = []
        self.feat_ids_list = []
        self.feat_ids_div = 5

        split_list = __C.SPLIT[RUN_MODE].split('+')
        for split in split_list:
            with open(__C.CAPTION_PATH[__C.DATASET][split + '-caps'], "r") as f:
                for line in f:
                    self.caps_list.append(line.strip())
            with open(__C.CAPTION_PATH[__C.DATASET][split + '-ids'], "r") as f:
                for line_i, line in enumerate(f):
                    if split in ['train']:
                        self.feat_ids_list.append(line.strip())
                    else:
                        if line_i % self.feat_ids_div == 0:
                            self.feat_ids_list.append(line.strip())
        print(' ========== Images size:', len(self.feat_ids_list))

        self.data_size = self.caps_list.__len__()
        print(' ========== Caption size:', self.data_size)

        id_map = None
        if __C.DATASET in ['flickr']:
            orin_caps = json.load(open(__C.CAPTION_PATH[__C.DATASET]['orin'], 'r'))
            id_map = {}
            for orin_cap in orin_caps['images']:
                orin_id = orin_cap['filename'].split('.')[0]
                id_map[orin_id] = str(orin_cap['imgid'])
            # print(id_map)


        # {image id} -> {image feature absolutely path}
        self.iid_to_frcn_feat_path = self.img_feat_path_load(frcn_feat_path_list, id_map=id_map)

        # Tokenize
        self.token_to_ix, self.pretrained_emb, max_token = self.tokenize(stat_caps_list, __C.GLOVE_FEATURE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Caption token vocab size:', self.token_size)

        if __C.MAX_TOKEN < 0:
            self.max_token = max_token
        else:
            self.max_token = __C.MAX_TOKEN
        print(' ========== Caption token max length:', self.max_token)

        self.neg_caps_idx_tensor = torch.randint(high=self.data_size, size=(len(self.feat_ids_list), self.__C.NEG_HARDSIZE)).long()
        self.neg_imgs_idx_tensor = torch.randint(high=len(self.feat_ids_list), size=(self.data_size, self.__C.NEG_HARDSIZE)).long()


    def shuffle_neg_idx(self):
        self.neg_caps_idx_tensor = torch.randint(high=self.data_size, size=(len(self.feat_ids_list), self.__C.NEG_HARDSIZE)).long()
        self.neg_imgs_idx_tensor = torch.randint(high=len(self.feat_ids_list), size=(self.data_size, self.__C.NEG_HARDSIZE)).long()


    def img_feat_path_load(self, path_list, id_map=None):
        iid_to_path = {}

        for ix, path in enumerate(path_list):
            iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
            # print(iid)
            if id_map is not None:
                if iid not in id_map:
                    continue
                iid = id_map[iid]
            iid_to_path[iid] = path

        return iid_to_path


    def tokenize(self, stat_caps_list, use_glove):
        max_token = 0
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

        for cap in stat_caps_list:
            words = re.sub(r"([.,'!?\"()*#:;])", '', cap.lower()).replace('-', ' ').replace('/', ' ').split()
            max_token = max(len(words), max_token)
            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb, max_token


    def get_all_caps(self):
        cap_ix_iter_list = []
        for idx in range(self.data_size):
            cap = self.caps_list[idx]
            cap_ix_iter = self.proc_cap(cap, self.token_to_ix, max_token=self.max_token)
            cap_ix_iter = torch.from_numpy(cap_ix_iter).unsqueeze(0)
            cap_ix_iter_list.append(cap_ix_iter)
        cap_ix_iter_list = torch.cat(cap_ix_iter_list, dim=0)

        return cap_ix_iter_list, torch.zeros(self.data_size, self.max_token, self.max_token, 3)


    def get_all_imgs(self):
        frcn_feat_iter_list = []
        bbox_feat_iter_list = []
        rel_img_iter_list = []
        for idx in range(len(self.feat_ids_list)):
            frcn_feat = np.load(self.iid_to_frcn_feat_path[self.feat_ids_list[idx]])
            frcn_feat_x = frcn_feat['x'].transpose((1, 0))
            frcn_feat_iter = self.proc_img_feat(frcn_feat_x, img_feat_pad_size=self.__C.FRCNFEAT_LEN)
            frcn_feat_iter = torch.from_numpy(frcn_feat_iter).unsqueeze(0)
            frcn_feat_iter_list.append(frcn_feat_iter)

            bbox_feat_iter = self.proc_img_feat(
                self.proc_bbox_feat(
                    frcn_feat['bbox'],
                    (frcn_feat['image_h'], frcn_feat['image_w'])
                ),
                img_feat_pad_size=self.__C.FRCNFEAT_LEN
            )
            bbox_feat_iter = torch.from_numpy(bbox_feat_iter).unsqueeze(0)
            bbox_feat_iter_list.append(bbox_feat_iter)

            bbox = torch.from_numpy(frcn_feat['bbox'].astype(np.float32))
            rel_img = relation_embedding(bbox)  # [n_obj, n_obj, 4]
            rel_img_iter = torch.zeros(self.__C.FRCNFEAT_LEN, self.__C.FRCNFEAT_LEN, 4)
            rel_img_iter[:rel_img.size(0), :rel_img.size(1), :] = rel_img[:]
            rel_img_iter = rel_img_iter.unsqueeze(0)
            rel_img_iter_list.append(rel_img_iter)

        frcn_feat_iter_list = torch.cat(frcn_feat_iter_list, dim=0)
        bbox_feat_iter_list = torch.cat(bbox_feat_iter_list, dim=0)
        rel_img_iter_list = torch.cat(rel_img_iter_list, dim=0)

        return frcn_feat_iter_list, bbox_feat_iter_list, rel_img_iter_list


    def __getitem__(self, idx):
        # pos caps
        cap = self.caps_list[idx]
        cap_ix_iter = self.proc_cap(cap, self.token_to_ix, max_token=self.max_token)
        cap_ix_iter = torch.from_numpy(cap_ix_iter)
        rel_cap_iter = torch.zeros(self.max_token, self.max_token, 3)

        # pos imgs
        frcn_feat = np.load(self.iid_to_frcn_feat_path[self.feat_ids_list[int(idx / self.feat_ids_div)]])
        frcn_feat_x = frcn_feat['x'].transpose((1, 0))
        frcn_feat_iter = self.proc_img_feat(frcn_feat_x, img_feat_pad_size=self.__C.FRCNFEAT_LEN)
        frcn_feat_iter = torch.from_numpy(frcn_feat_iter)


        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['image_h'], frcn_feat['image_w'])
            ),
            img_feat_pad_size=self.__C.FRCNFEAT_LEN
        )
        bbox_feat_iter = torch.from_numpy(bbox_feat_iter)

        bbox = torch.from_numpy(frcn_feat['bbox'].astype(np.float32))
        rel_img = relation_embedding(bbox)  # [n_obj, n_obj, 4]
        rel_img_iter = torch.zeros(self.__C.FRCNFEAT_LEN, self.__C.FRCNFEAT_LEN, 4)
        rel_img_iter[:rel_img.size(0), :rel_img.size(1), :] = rel_img[:]

        neg_frcn_feat_iter = torch.zeros(1)
        neg_bbox_feat_iter = torch.zeros(1)
        neg_rel_img_iter = torch.zeros(1)
        neg_cap_ix_iter = torch.zeros(1)
        neg_rel_cap_iter = torch.zeros(1)
        if self.RUN_MODE == 'train':
            # neg caps
            neg_cap_idx = random.randint(0, self.__C.NEG_HARDSIZE - 1)
            neg_cap = self.caps_list[self.neg_caps_idx_tensor[int(idx / self.feat_ids_div), neg_cap_idx]]
            neg_cap_ix_iter = self.proc_cap(neg_cap, self.token_to_ix, max_token=self.max_token)
            neg_cap_ix_iter = torch.from_numpy(neg_cap_ix_iter)
            neg_rel_cap_iter = torch.zeros(self.max_token, self.max_token, 3)

            # neg imgs
            neg_img_idx = random.randint(0, self.__C.NEG_HARDSIZE - 1)
            neg_frcn_feat = np.load(self.iid_to_frcn_feat_path[self.feat_ids_list[self.neg_imgs_idx_tensor[idx, neg_img_idx]]])
            neg_frcn_feat_x = neg_frcn_feat['x'].transpose((1, 0))
            neg_frcn_feat_iter = self.proc_img_feat(neg_frcn_feat_x, img_feat_pad_size=self.__C.FRCNFEAT_LEN)
            neg_frcn_feat_iter = torch.from_numpy(neg_frcn_feat_iter)

            neg_bbox_feat_iter = self.proc_img_feat(
                self.proc_bbox_feat(
                    neg_frcn_feat['bbox'],
                    (neg_frcn_feat['image_h'], neg_frcn_feat['image_w'])
                ),
                img_feat_pad_size=self.__C.FRCNFEAT_LEN
            )
            neg_bbox_feat_iter = torch.from_numpy(neg_bbox_feat_iter)

            neg_bbox = torch.from_numpy(neg_frcn_feat['bbox'].astype(np.float32))
            neg_rel_img = relation_embedding(neg_bbox)  # [n_obj, n_obj, 4]
            neg_rel_img_iter = torch.zeros(self.__C.FRCNFEAT_LEN, self.__C.FRCNFEAT_LEN, 4)
            neg_rel_img_iter[:neg_rel_img.size(0), :neg_rel_img.size(1), :] = neg_rel_img[:]


        return frcn_feat_iter, \
               bbox_feat_iter, \
               rel_img_iter, \
               cap_ix_iter, \
               rel_cap_iter,\
               neg_frcn_feat_iter, \
               neg_bbox_feat_iter, \
               neg_rel_img_iter, \
               neg_cap_ix_iter, \
               neg_rel_cap_iter

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


    def proc_cap(self, cap, token_to_ix, max_token):
        cap_ix = np.zeros(max_token, np.int64)
        words = re.sub(r"([.,'!?\"()*#:;])", '', cap.lower()).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                cap_ix[ix] = token_to_ix[word]
            else:
                cap_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return cap_ix



class DataSet_Neg(Data.Dataset):
    def __init__(self, __C, keep):
        print(' ========== Neg Loader Keep:', keep)
        self.__C = __C
        self.keep = keep

        # Loading all image paths
        frcn_feat_path_list = []
        for feat_split in __C.IMGFEAT_PATH[__C.DATASET]:
            frcn_feat_path_list += glob.glob(__C.IMGFEAT_PATH[__C.DATASET][feat_split] + '*.npz')

        # Loading question word list
        stat_caps_list = []
        for cap_split in __C.CAPTION_PATH[__C.DATASET]:
            if 'caps' in cap_split:
                with open(__C.CAPTION_PATH[__C.DATASET][cap_split], "r") as f:
                    for line in f:
                        # print(line)
                        stat_caps_list.append(line.strip())


        # Loading question and answer list
        self.caps_list = []
        self.feat_ids_list = []
        self.feat_ids_div = 5

        split_list = __C.SPLIT['train'].split('+')
        for split in split_list:
            with open(__C.CAPTION_PATH[__C.DATASET][split + '-caps'], "r") as f:
                for line in f:
                    self.caps_list.append(line.strip())
            with open(__C.CAPTION_PATH[__C.DATASET][split + '-ids'], "r") as f:
                for line_i, line in enumerate(f):
                    if split in ['train']:
                        self.feat_ids_list.append(line.strip())
                    else:
                        if line_i % self.feat_ids_div == 0:
                            self.feat_ids_list.append(line.strip())
        print(' ========== Images size:', len(self.feat_ids_list))

        self.data_size = self.caps_list.__len__()
        print(' ========== Caption size:', self.data_size)

        id_map = None
        if __C.DATASET in ['flickr']:
            orin_caps = json.load(open(__C.CAPTION_PATH[__C.DATASET]['orin'], 'r'))
            id_map = {}
            for orin_cap in orin_caps['images']:
                orin_id = orin_cap['filename'].split('.')[0]
                id_map[orin_id] = str(orin_cap['imgid'])
            # print(id_map)

        # {image id} -> {image feature absolutely path}
        self.iid_to_frcn_feat_path = self.img_feat_path_load(frcn_feat_path_list, id_map=id_map)

        # Tokenize
        self.token_to_ix, self.pretrained_emb, max_token = self.tokenize(stat_caps_list, __C.GLOVE_FEATURE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Caption token vocab size:', self.token_size)

        if __C.MAX_TOKEN < 0:
            self.max_token = max_token
        else:
            self.max_token = __C.MAX_TOKEN
        print(' ========== Caption token max length:', self.max_token)


    def img_feat_path_load(self, path_list, id_map=None):
        iid_to_path = {}

        for ix, path in enumerate(path_list):
            iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
            # print(iid)
            if id_map is not None:
                if iid not in id_map:
                    continue
                iid = id_map[iid]
            iid_to_path[iid] = path

        return iid_to_path


    def tokenize(self, stat_caps_list, use_glove):
        max_token = 0
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

        for cap in stat_caps_list:
            words = re.sub(r"([.,'!?\"()*#:;])", '', cap.lower()).replace('-', ' ').replace('/', ' ').split()
            max_token = max(len(words), max_token)
            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb, max_token


    def get_all_caps(self):
        cap_ix_iter_list = []
        for idx in range(self.data_size):
            cap = self.caps_list[idx]
            cap_ix_iter = self.proc_cap(cap, self.token_to_ix, max_token=self.max_token)
            cap_ix_iter = torch.from_numpy(cap_ix_iter).unsqueeze(0)
            cap_ix_iter_list.append(cap_ix_iter)
        cap_ix_iter_list = torch.cat(cap_ix_iter_list, dim=0)

        return cap_ix_iter_list, torch.zeros(self.data_size, self.max_token, self.max_token, 3)


    def get_all_imgs(self):
        frcn_feat_iter_list = []
        bbox_feat_iter_list = []
        rel_img_iter_list = []
        for idx in range(len(self.feat_ids_list)):
            frcn_feat = np.load(self.iid_to_frcn_feat_path[self.feat_ids_list[idx]])
            frcn_feat_x = frcn_feat['x'].transpose((1, 0))
            frcn_feat_iter = self.proc_img_feat(frcn_feat_x, img_feat_pad_size=self.__C.FRCNFEAT_LEN)
            frcn_feat_iter = torch.from_numpy(frcn_feat_iter).unsqueeze(0)
            frcn_feat_iter_list.append(frcn_feat_iter)

            bbox_feat_iter = self.proc_img_feat(
                self.proc_bbox_feat(
                    frcn_feat['bbox'],
                    (frcn_feat['image_h'], frcn_feat['image_w'])
                ),
                img_feat_pad_size=self.__C.FRCNFEAT_LEN
            )
            bbox_feat_iter = torch.from_numpy(bbox_feat_iter).unsqueeze(0)
            bbox_feat_iter_list.append(bbox_feat_iter)

            bbox = torch.from_numpy(frcn_feat['bbox'].astype(np.float32))
            rel_img = relation_embedding(bbox)  # [n_obj, n_obj, 4]
            rel_img_iter = torch.zeros(self.__C.FRCNFEAT_LEN, self.__C.FRCNFEAT_LEN, 4)
            rel_img_iter[:rel_img.size(0), :rel_img.size(1), :] = rel_img[:]
            rel_img_iter = rel_img_iter.unsqueeze(0)
            rel_img_iter_list.append(rel_img_iter)

        frcn_feat_iter_list = torch.cat(frcn_feat_iter_list, dim=0)
        bbox_feat_iter_list = torch.cat(bbox_feat_iter_list, dim=0)
        rel_img_iter_list = torch.cat(rel_img_iter_list, dim=0)

        return frcn_feat_iter_list, bbox_feat_iter_list, rel_img_iter_list


    def __getitem__(self, idx):
        if self.keep == 'caps':
            cap = self.caps_list[idx]
            cap_ix_iter = self.proc_cap(cap, self.token_to_ix, max_token=self.max_token)
            cap_ix_iter_list = torch.from_numpy(cap_ix_iter).unsqueeze(0).repeat(self.__C.NEG_RANDSIZE, 1)
            rel_cap_iter_list = torch.zeros(self.__C.NEG_RANDSIZE, self.max_token, self.max_token, 3)

            frcn_feat_iter_list = torch.zeros(self.__C.NEG_RANDSIZE, self.__C.FRCNFEAT_LEN, self.__C.FRCNFEAT_SIZE)
            bbox_feat_iter_list = torch.zeros(self.__C.NEG_RANDSIZE, self.__C.FRCNFEAT_LEN, 5)
            rel_img_iter_list = torch.zeros(self.__C.NEG_RANDSIZE, self.__C.FRCNFEAT_LEN, self.__C.FRCNFEAT_LEN, 4)
            neg_idx_list = torch.zeros(self.__C.NEG_RANDSIZE).long()
            for neg_step in range(self.__C.NEG_RANDSIZE):
                avoid_id = int(idx / self.feat_ids_div)
                rind = random.randint(0, len(self.feat_ids_list) - 1)
                while (rind == avoid_id):
                    rind = random.randint(0, len(self.feat_ids_list) - 1)
                neg_idx_list[neg_step] = rind

        else:
            frcn_feat = np.load(self.iid_to_frcn_feat_path[self.feat_ids_list[idx]])
            frcn_feat_x = frcn_feat['x'].transpose((1, 0))
            frcn_feat_iter = self.proc_img_feat(frcn_feat_x, img_feat_pad_size=self.__C.FRCNFEAT_LEN)
            frcn_feat_iter_list = torch.from_numpy(frcn_feat_iter).unsqueeze(0).repeat(self.__C.NEG_RANDSIZE, 1, 1)

            bbox_feat_iter = self.proc_img_feat(
                self.proc_bbox_feat(
                    frcn_feat['bbox'],
                    (frcn_feat['image_h'], frcn_feat['image_w'])
                ),
                img_feat_pad_size=self.__C.FRCNFEAT_LEN
            )
            bbox_feat_iter_list = torch.from_numpy(bbox_feat_iter).unsqueeze(0).repeat(self.__C.NEG_RANDSIZE, 1, 1)

            bbox = torch.from_numpy(frcn_feat['bbox'].astype(np.float32))
            rel_img = relation_embedding(bbox)  # [n_obj, n_obj, 4]
            rel_img_iter = torch.zeros(self.__C.FRCNFEAT_LEN, self.__C.FRCNFEAT_LEN, 4)
            rel_img_iter[:rel_img.size(0), :rel_img.size(1), :] = rel_img[:]
            rel_img_iter_list = rel_img_iter.unsqueeze(0).repeat(self.__C.NEG_RANDSIZE, 1, 1, 1)

            cap_ix_iter_list = torch.zeros(self.__C.NEG_RANDSIZE, self.max_token).long()
            rel_cap_iter_list = torch.zeros(self.__C.NEG_RANDSIZE, self.max_token, self.max_token, 3)
            neg_idx_list = torch.zeros(self.__C.NEG_RANDSIZE).long()
            for neg_step in range(self.__C.NEG_RANDSIZE):
                avoid_id = tuple(idx * self.feat_ids_div + bias for bias in range(self.feat_ids_div))
                rind = random.randint(0, self.data_size - 1)
                while (rind in avoid_id):
                    rind = random.randint(0, self.data_size - 1)
                neg_idx_list[neg_step] = rind

                # print(rind)

                cap = self.caps_list[rind]
                cap_ix_iter = self.proc_cap(cap, self.token_to_ix, max_token=self.max_token)
                cap_ix_iter = torch.from_numpy(cap_ix_iter)
                # print(cap_ix_iter.size())
                cap_ix_iter_list[neg_step, :] = cap_ix_iter

        return frcn_feat_iter_list, bbox_feat_iter_list, rel_img_iter_list, cap_ix_iter_list, rel_cap_iter_list, neg_idx_list


    def __len__(self):
        if self.keep == 'caps':
            return self.data_size
        else:
            return len(self.feat_ids_list)


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


    def proc_cap(self, cap, token_to_ix, max_token):
        cap_ix = np.zeros(max_token, np.int64)
        words = re.sub(r"([.,'!?\"()*#:;])", '', cap.lower()).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                cap_ix[ix] = token_to_ix[word]
            else:
                cap_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return cap_ix
