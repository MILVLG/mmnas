import math, os, json, torch, datetime, random, copy, shutil, torchvision
# import sys
# sys.path.append('../..')

import torch.nn as nn
import torch.optim as Optim
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from collections import namedtuple
from tkinter import _flatten

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from mmnas.loader.load_data_itm import DataSet, DataSet_Neg
from mmnas.loader.filepath_itm import Path
from mmnas.model.hygr_itm import Net_Search
from mmnas.utils.optimizer import WarmupOptimizer
from mmnas.utils.sampler import SubsetDistributedSampler
from mmnas.model.mixed import MixedOp
from mmnas.utils.itm_loss import BCE_Loss, Margin_Loss

MASTER_ADDR = 'localhost'
MASTER_PORT = '1240'

GPU = '0, 1, 2, 3'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
# torch.set_num_threads(2)
WORLD_SIZE = len(GPU.split(','))
# WORLD_SIZE = 4
VERSION = 'train_itm'

RUN_MODE = 'train'
# RUN_MODE = 'val'
# RUN_MODE = 'test'

class CfgSearch(Path):
    def __init__(self, rank, world_size):
        super(CfgSearch, self).__init__()

        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = MASTER_PORT
        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        self.DEBUG = True
        # self.DEBUG = False

        # Set Devices
        self.WORLD_SIZE = world_size
        self.RANK = rank
        self.N_GPU = torch.cuda.device_count() // self.WORLD_SIZE
        self.DEVICE_IDS = list(range(self.RANK * self.N_GPU, (self.RANK + 1) * self.N_GPU))

        # Set Seed For CPU And GPUs
        self.SEED = 888
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)
        np.random.seed(self.SEED)
        # random.seed(self.SEED)
        random.seed(self.SEED * self.RANK)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        # Version Control
        self.VERSION = VERSION + '-search'
        self.RESUME = False
        self.CKPT_FILE_PATH = None
        self.CKPT_EPOCH = 0

        # self.DATASET = 'coco'
        self.DATASET = 'flickr'
        self.SPLIT = {
            'train': 'train',
            'val': 'dev',
            'test': 'test',
        }
        self.SPLIT_PORTION = 0.8

        self.NUM_WORKERS = 4
        self.BATCH_SIZE = 64
        self.EVAL_BATCH_SIZE = self.BATCH_SIZE

        self.NEG_BATCHSIZE = 128
        self.NEG_RANDSIZE = 32
        self.NEG_HARDSIZE = 5
        self.NEG_NEPOCH = 1
        self.NEG_START_EPOCH = 10

        self.BBOX_FEATURE = False
        self.FRCNFEAT_LEN = 36
        self.FRCNFEAT_SIZE = 2048
        self.BBOXFEAT_EMB_SIZE = 2048
        self.GLOVE_FEATURE = True
        self.WORD_EMBED_SIZE = 300
        self.MAX_TOKEN = 50
        self.REL_SIZE = 64

        # Network Params
        self.LAYERS = 1
        self.NODES = {
            'enc': 12,
            'dec': 18,
        }
        self.HSIZE = 256
        self.DROPOUT_R = 0.1
        self.OPS_RESIDUAL = True
        self.OPS_NORM = True

        self.ATTFLAT_GLIMPSES = 1
        self.ATTFLAT_OUT_SIZE = self.HSIZE * 2
        self.ATTFLAT_MLP_SIZE = 512

        self.SCORES_LOSS = 'bce'
        # self.SCORES_LOSS = 'margin'
        # self.MAX_VIOLATION = True
        # self.MAX_VIOLATION = False

        # Optimizer Params
        # self.NET_OPTIM = 'sgd'
        self.NET_OPTIM = 'wadam'
        self.REDUCTION = 'sum'
        # self.REDUCTION = 'mean'

        if self.NET_OPTIM == 'sgd':
            self.NET_LR_BASE = 0.005
            self.NET_LR_MIN = 0.0005
            self.NET_MOMENTUM = 0.9
            # self.NET_WEIGHT_DECAY = 1e-4
            self.NET_WEIGHT_DECAY = 0
            self.NET_GRAD_CLIP = 5.  # GRAD_CLIP = -1: means not use grad_norm_clip 0.01
            # self.NET_GRAD_CLIP = 1.  # GRAD_CLIP = -1: means not use grad_norm_clip 0.05
            # self.NET_GRAD_CLIP = -1  # GRAD_CLIP = -1: means not use grad_norm_clip
            self.MAX_EPOCH = 50

        else:
            self.NET_OPTIM_WARMUP = True
            self.NET_LR_BASE = 0.0001
            # self.NET_LR_BASE = 0.0002
            # self.NET_WEIGHT_DECAY = 1e-5
            self.NET_WEIGHT_DECAY = 0
            self.NET_GRAD_CLIP = 1.  # GRAD_CLIP = -1: means not use grad_norm_clip
            # self.NET_GRAD_CLIP = -1  # GRAD_CLIP = -1: means not use grad_norm_clip
            self.NET_LR_DECAY_R = 0.2
            # self.NET_LR_DECAY_LIST = [10, 12]
            self.NET_LR_DECAY_LIST = []
            self.OPT_BETAS = (0.9, 0.98)
            self.OPT_EPS = 1e-9
            self.MAX_EPOCH = 200

        self.ALPHA_START = 20
        self.ALPHA_EVERY = 5
        # self.ALPHA_BINARY_MODE = 'full_v2'
        self.ALPHA_BINARY_MODE = 'full'
        # self.ALPHA_BINARY_MODE = 'two'
        # self.ALPHA_LR_BASE = 1.
        self.ALPHA_LR_BASE = 0.1
        # self.ALPHA_WEIGHT_DECAY = 1e-3
        self.ALPHA_WEIGHT_DECAY = 0
        self.ALPHA_INIT_TYPE = 'normal'
        # self.ALPHA_INIT_TYPE = 'uniform'
        # self.ALPHA_OPT_BETAS = (0.5, 0.999)
        self.ALPHA_OPT_BETAS = (0., 0.999)

        # self.OPS_ADAPTER = OpsAdapter()
        self.GENOTYPES_K = 1
        # self.REDUMP_EVAL = False
        self.REDUMP_EVAL = True



class Execution:
    def __init__(self, __C):
        self.__C = __C

    def get_optim(self, net, search=False, epoch_steps=None):
        if self.__C.NET_OPTIM == 'sgd':
            net_optim = torch.optim.SGD(net.module.net_parameters() if search else net.parameters(), self.__C.NET_LR_BASE, momentum=self.__C.NET_MOMENTUM,
                                        weight_decay=self.__C.NET_WEIGHT_DECAY)
        else:
            net_optim = WarmupOptimizer(
                self.__C.NET_LR_BASE,
                Optim.Adam(
                    # filter(lambda p: p.requires_grad, net.parameters()),
                    net.module.net_parameters() if search else net.parameters(),
                    lr=0,
                    betas=self.__C.OPT_BETAS,
                    eps=self.__C.OPT_EPS,
                    weight_decay=self.__C.NET_WEIGHT_DECAY,
                ),
                epoch_steps,
                warmup=self.__C.NET_OPTIM_WARMUP,
            )


        alpha_optim = torch.optim.Adam(net.module.alpha_prob_parameters(), self.__C.ALPHA_LR_BASE, betas=self.__C.ALPHA_OPT_BETAS,
                                           weight_decay=self.__C.ALPHA_WEIGHT_DECAY)

        return net_optim, alpha_optim


    def search(self, train_loader, search_loader, eval_loader, neg_caps_loader, neg_imgs_loader):
        # data_size = train_loader.sampler.total_size
        init_dict = {
            'token_size': train_loader.dataset.token_size,
            'pretrained_emb': train_loader.dataset.pretrained_emb,
        }

        net = Net_Search(self.__C, init_dict)
        net.to(self.__C.DEVICE_IDS[0])
        net = DDP(net, device_ids=self.__C.DEVICE_IDS)
        if self.__C.SCORES_LOSS in ['bce']:
            loss_fn = BCE_Loss(self.__C)
        else:
            loss_fn = Margin_Loss(self.__C)

        if self.__C.RESUME:
            print(' ========== Resume training')
            path = self.__C.CKPT_FILE_PATH
            print('Loading the {}'.format(path))

            rank0_devices = [x - self.__C.RANK * len(self.__C.DEVICE_IDS) for x in self.__C.DEVICE_IDS]
            device_pairs = zip(rank0_devices, self.__C.DEVICE_IDS)
            map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
            ckpt = torch.load(path, map_location=map_location)
            print('Finish loading ckpt !!!')
            net.load_state_dict(ckpt['state_dict'])

            lr_scheduler = None
            start_epoch = self.__C.CKPT_EPOCH
            net_optim, alpha_optim = self.get_optim(net, search=True, epoch_steps=len(train_loader))
            if self.__C.NET_OPTIM == 'sgd':
                net_optim.load_state_dict(ckpt['net_optim'])
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    net_optim, self.__C.MAX_EPOCH, eta_min=self.__C.NET_LR_MIN, last_epoch=start_epoch)
            else:
                net_optim.optimizer.load_state_dict(ckpt['net_optim'])
                net_optim.set_start_step(start_epoch * len(train_loader))
            # alpha_optim.load_state_dict(ckpt['alpha_optim'])

        else:
            net_optim, alpha_optim = self.get_optim(net, search=True, epoch_steps=len(train_loader))
            start_epoch = 0

            lr_scheduler = None
            if self.__C.NET_OPTIM == 'sgd':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    net_optim, self.__C.MAX_EPOCH, eta_min=self.__C.NET_LR_MIN)

        loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        print('loading all images ...')
        all_frcn_feat_iter_list, all_bbox_feat_iter_list, all_rel_img_iter_list = neg_imgs_loader.dataset.get_all_imgs()
        print('loading finished')
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):
            if self.__C.RANK == 0:
                logfile = open('./logs/log/log_' + self.__C.VERSION + '.txt', 'a+')
                logfile.write('nowTime: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
                logfile.close()

            # ================================  NEG  ================================
            if epoch % self.__C.NEG_NEPOCH == 0 and epoch >= self.__C.NEG_START_EPOCH:
                net.eval()
                MixedOp.MODE = None
                net.module.reset_binary_gates()
                net.module.unused_modules_off()

                with torch.no_grad():
                    # neg_caps_idx_dict
                    dist.barrier()
                    print('reset negative captions ...')
                    neg_caps_idx_list = []
                    for step, (frcn_feat_iter_list, bbox_feat_iter_list, rel_img_iter_list, cap_ix_iter_list, rel_cap_iter_list, neg_idx_list) in enumerate(neg_caps_loader):
                        if step % 10 == 0 and self.__C.RANK == 0:
                            print('negative captions percent', step / len(neg_caps_loader) * 100.)

                        frcn_feat_iter_list = frcn_feat_iter_list.view(-1, self.__C.FRCNFEAT_LEN, self.__C.FRCNFEAT_SIZE)
                        bbox_feat_iter_list = bbox_feat_iter_list.view(-1, self.__C.FRCNFEAT_LEN, 5)
                        rel_img_iter_list = rel_img_iter_list.view(-1, self.__C.FRCNFEAT_LEN, self.__C.FRCNFEAT_LEN, 4)
                        cap_ix_iter_list = cap_ix_iter_list.view(-1, neg_caps_loader.dataset.max_token)
                        rel_cap_iter_list = rel_cap_iter_list.view(-1, neg_caps_loader.dataset.max_token, neg_caps_loader.dataset.max_token, 3)

                        input = (frcn_feat_iter_list, bbox_feat_iter_list, rel_img_iter_list, cap_ix_iter_list, rel_cap_iter_list)
                        scores = net(input)
                        scores = scores.view(-1, self.__C.NEG_RANDSIZE)
                        arg_scores = torch.argsort(scores, dim=-1, descending=True)[:, :self.__C.NEG_HARDSIZE]
                        arg_scores_bi = torch.arange(arg_scores.size(0)).unsqueeze(1).expand_as(arg_scores)
                        scores_ind = neg_idx_list[arg_scores_bi, arg_scores].to(scores.device)
                        neg_caps_idx_list.append(scores_ind)

                    neg_caps_idx_list = torch.cat(neg_caps_idx_list, dim=0)
                    neg_caps_idx_list_gather = [torch.zeros_like(neg_caps_idx_list.unsqueeze(1)) for _ in
                                                range(self.__C.WORLD_SIZE)]
                    dist.all_gather(neg_caps_idx_list_gather, neg_caps_idx_list.unsqueeze(1))
                    neg_caps_idx_list_gather = torch.cat(neg_caps_idx_list_gather, dim=1).view(-1,
                                                                                               self.__C.NEG_HARDSIZE).cpu()  # torch.Size([29000, 20])

                    rest_caps_num = neg_caps_loader.sampler.rest_data_num
                    if rest_caps_num:
                        neg_caps_idx_list_gather = neg_caps_idx_list_gather[: -rest_caps_num]
                    train_loader.dataset.neg_caps_idx_tensor = neg_caps_idx_list_gather
                    search_loader.dataset.neg_caps_idx_tensor = neg_caps_idx_list_gather

                    # neg_imgs_idx_dict
                    dist.barrier()
                    print('reset negative images ...')
                    neg_imgs_idx_list = []
                    for step, (frcn_feat_iter_list, bbox_feat_iter_list, rel_img_iter_list, cap_ix_iter_list, rel_cap_iter_list, neg_idx_list) in enumerate(neg_imgs_loader):
                        if step % 10 == 0 and self.__C.RANK == 0:
                            print('negative images percent', step / len(neg_imgs_loader) * 100.)

                        frcn_feat_iter_list = all_frcn_feat_iter_list[neg_idx_list, :]
                        bbox_feat_iter_list = all_bbox_feat_iter_list[neg_idx_list, :]
                        rel_img_iter_list = all_rel_img_iter_list[neg_idx_list, :]
                        frcn_feat_iter_list = frcn_feat_iter_list.view(-1, self.__C.FRCNFEAT_LEN, self.__C.FRCNFEAT_SIZE)
                        bbox_feat_iter_list = bbox_feat_iter_list.view(-1, self.__C.FRCNFEAT_LEN, 5)
                        rel_img_iter_list = rel_img_iter_list.view(-1, self.__C.FRCNFEAT_LEN, self.__C.FRCNFEAT_LEN, 4)

                        cap_ix_iter_list = cap_ix_iter_list.view(-1, neg_caps_loader.dataset.max_token)
                        rel_cap_iter_list = rel_cap_iter_list.view(-1, neg_caps_loader.dataset.max_token, neg_caps_loader.dataset.max_token, 3)
                        input = (frcn_feat_iter_list, bbox_feat_iter_list, rel_img_iter_list, cap_ix_iter_list, rel_cap_iter_list)

                        scores = net(input)
                        scores = scores.view(-1, self.__C.NEG_RANDSIZE)
                        arg_scores = torch.argsort(scores, dim=-1, descending=True)[:, :self.__C.NEG_HARDSIZE]
                        arg_scores_bi = torch.arange(arg_scores.size(0)).unsqueeze(1).expand_as(arg_scores)
                        scores_ind = neg_idx_list[arg_scores_bi, arg_scores].to(scores.device)
                        neg_imgs_idx_list.append(scores_ind)

                    neg_imgs_idx_list = torch.cat(neg_imgs_idx_list, dim=0)
                    neg_imgs_idx_list_gather = [torch.zeros_like(neg_imgs_idx_list.unsqueeze(1)) for _ in range(self.__C.WORLD_SIZE)]
                    dist.all_gather(neg_imgs_idx_list_gather, neg_imgs_idx_list.unsqueeze(1))
                    neg_imgs_idx_list_gather = torch.cat(neg_imgs_idx_list_gather, dim=1).view(-1, self.__C.NEG_HARDSIZE).cpu()  # torch.Size([145000, 20])

                    rest_imgs_num = neg_imgs_loader.sampler.rest_data_num
                    if rest_imgs_num:
                        neg_imgs_idx_list_gather = neg_imgs_idx_list_gather[: -rest_imgs_num]
                    # print(neg_imgs_idx_list_gather.size())
                    # train_loader.dataset.neg_imgs_idx_dict = {imgs_step: imgs_ind for imgs_step, imgs_ind in enumerate(neg_imgs_idx_list_gather)}
                    train_loader.dataset.neg_imgs_idx_tensor = neg_imgs_idx_list_gather
                    search_loader.dataset.neg_imgs_idx_tensor = neg_imgs_idx_list_gather

                net.module.unused_modules_back()

            elif epoch < self.__C.NEG_START_EPOCH:
                print('shuffle neg idx')
                train_loader.dataset.shuffle_neg_idx()
                search_loader.dataset.shuffle_neg_idx()

            # ================================  NEG  ================================

            print('Training Epoch:', epoch)
            train_loader.sampler.set_epoch(epoch)
            search_loader.sampler.set_epoch(epoch)
            search_loader.sampler.set_shuffle(True)
            net.train()

            if self.__C.NET_OPTIM == 'sgd':
                lr_scheduler.step()
            else:
                if epoch in self.__C.NET_LR_DECAY_LIST:
                    net_optim.decay(self.__C.NET_LR_DECAY_R)


            eval_iter = iter(search_loader)
            for step, (train_frcn_feat, train_bbox_feat, train_rel_img_iter, train_cap_ix, train_rel_cap_iter,
                       train_neg_frcn_feat, train_neg_bbox_feat, train_neg_rel_img_iter, train_neg_cap_ix, train_neg_rel_cap_iter) in enumerate(train_loader):
                train_input_pos = (train_frcn_feat, train_bbox_feat, train_rel_img_iter, train_cap_ix, train_rel_cap_iter)
                train_input_negc = (train_frcn_feat, train_bbox_feat, train_rel_img_iter, train_neg_cap_ix, train_neg_rel_cap_iter)
                train_input_negi = (train_neg_frcn_feat, train_neg_bbox_feat, train_neg_rel_img_iter, train_cap_ix, train_rel_cap_iter)

                if (step + 1) % 100 == 0 and self.__C.RANK == 0:
                    print(net.module.genotype())
                    print(net.module.genotype_weights())

                # network step
                MixedOp.MODE = None
                net.module.reset_binary_gates()
                net.module.unused_modules_off()
                scores_pos = net(train_input_pos)
                scores_negc = net(train_input_negc)
                scores_negi = net(train_input_negi)
                loss = loss_fn(scores_pos, scores_negc, scores_negi)

                # for avoid backward the unused params
                loss += 0 * sum(p.sum() for p in net.module.alpha_prob_parameters())
                loss += 0 * sum(p.sum() for p in net.module.alpha_gate_parameters())
                loss += 0 * sum(p.sum() for p in net.module.net_parameters())

                # net_optim.zero_grad()
                net.zero_grad()
                loss.backward()
                loss_sum += loss.item()

                # gradient clipping
                if self.__C.NET_GRAD_CLIP > 0:
                    # nn.utils.clip_grad_norm_(net.parameters(), self.__C.NET_GRAD_CLIP)
                    nn.utils.clip_grad_norm_(net.module.net_parameters(), self.__C.NET_GRAD_CLIP)

                net_optim.step()
                net.module.unused_modules_back()

                # Arch Params Updating
                is_arch_step = False
                if epoch >= self.__C.ALPHA_START and (step + 1) % self.__C.ALPHA_EVERY == 0:
                    is_arch_step = True
                if is_arch_step:
                    try:
                        eval_frcn_feat, eval_bbox_feat, eval_rel_img_iter, eval_cap_ix, eval_rel_cap_iter, \
                        eval_neg_frcn_feat, eval_neg_bbox_feat, eval_neg_rel_img_iter, eval_neg_cap_ix, eval_neg_rel_cap_iter = next(eval_iter)
                    except StopIteration:
                        eval_iter = iter(search_loader)
                        eval_frcn_feat, eval_bbox_feat, eval_rel_img_iter, eval_cap_ix, eval_rel_cap_iter, \
                        eval_neg_frcn_feat, eval_neg_bbox_feat, eval_neg_rel_img_iter, eval_neg_cap_ix, eval_neg_rel_cap_iter = next(eval_iter)

                    eval_input_pos = (eval_frcn_feat, eval_bbox_feat, eval_rel_img_iter, eval_cap_ix, eval_rel_cap_iter)
                    eval_input_negc = (eval_frcn_feat, eval_bbox_feat, eval_rel_img_iter, eval_neg_cap_ix, eval_neg_rel_cap_iter)
                    eval_input_negi = (eval_neg_frcn_feat, eval_neg_bbox_feat, eval_neg_rel_img_iter, eval_cap_ix, eval_rel_cap_iter)

                    MixedOp.MODE = self.__C.ALPHA_BINARY_MODE
                    net.module.reset_binary_gates()
                    net.module.unused_modules_off()
                    scores_pos = net(eval_input_pos)
                    scores_negc = net(eval_input_negc)
                    scores_negi = net(eval_input_negi)
                    loss = loss_fn(scores_pos, scores_negc, scores_negi)

                    # for avoid backward the unused params
                    loss += 0 * sum(p.sum() for p in net.module.alpha_prob_parameters())
                    # loss += 0 * sum(p.sum() for p in net.module.alpha_gate_parameters())
                    loss += 0 * sum(p.sum() for p in net.module.net_parameters())

                    # alpha_optim.zero_grad()
                    net.zero_grad()
                    loss.backward()
                    net.module.set_arch_param_grad()
                    alpha_optim.step()

                    if MixedOp.MODE == 'two':
                        net.module.rescale_updated_arch_param()
                    net.module.unused_modules_back()
                    MixedOp.MODE = None

                if self.__C.DEBUG and self.__C.RANK == 0:
                    if self.__C.REDUCTION == 'sum':
                        print(step, loss.item() / self.__C.BATCH_SIZE)
                    else:
                        print(step, loss.item())

            # ======== Per Epoch Finish ========
            epoch_finish = epoch + 1

            if self.__C.RANK == 0:
                state = {
                    'state_dict': net.state_dict(),
                    'net_optim': net_optim.state_dict() if self.__C.NET_OPTIM == 'sgd' else net_optim.optimizer.state_dict(),
                    'alpha_optim': alpha_optim.state_dict(),
                }
                torch.save(state, self.__C.CKPT_PATH + self.__C.VERSION + '_epoch' + str(epoch_finish) + '.pkl')

                if self.__C.NET_OPTIM == 'sgd':
                    lr_cur = lr_scheduler.get_lr()[0]
                else:
                    lr_cur = net_optim._rate

                genotype = net.module.genotype()
                genotype_weights = net.module.genotype_weights()
                logfile = open('./logs/log/log_' + self.__C.VERSION + '.txt', 'a+')
                if self.__C.REDUCTION == 'sum':
                    logfile.write('epoch = ' + str(epoch_finish) + '  loss = ' +
                                  str(loss_sum / len(train_loader) / self.__C.BATCH_SIZE) +
                                  '\n' + 'lr = ' + str(lr_cur) + '\n')
                else:
                    logfile.write('epoch = ' + str(epoch_finish) + '  loss = ' + str(loss_sum / len(train_loader)) +
                                  '\n' + 'lr = ' + str(lr_cur) + '\n')

                for genotype_name in genotype:
                    logfile.write(genotype_name + ': ' + str(genotype[genotype_name]) + '\n')
                    print(genotype_name + ': ' + str(genotype[genotype_name]))
                for genotype_name in genotype_weights:
                    logfile.write(genotype_name + ': ' + str(genotype_weights[genotype_name]) + '\n')
                    print(genotype_name + ': ' + str(genotype_weights[genotype_name]))
                logfile.close()

                if epoch_finish == 1 + start_epoch:
                    json.dump({}, open(self.__C.EVAL_PATH['arch'] + self.__C.VERSION + '.json', 'w+'))
                genotype_json = json.load(open(self.__C.EVAL_PATH['arch'] + self.__C.VERSION + '.json', 'r+'))
                genotype_json['epoch' + str(epoch_finish)] = genotype
                json.dump(genotype_json, open(self.__C.EVAL_PATH['arch'] + self.__C.VERSION + '.json', 'w+'))

            dist.barrier()

            if eval_loader is not None:
                self.eval(
                    eval_loader,
                    net=net,
                    valid=True,
                    redump=self.__C.REDUMP_EVAL and epoch_finish == 1 + start_epoch
                )
            loss_sum = 0


    def eval(self, eval_loader, net=None, valid=False, redump=False):
        init_dict = {
            'token_size': eval_loader.dataset.token_size,
            'pretrained_emb': eval_loader.dataset.pretrained_emb,
        }

        if net is None:
            rank0_devices = [x - self.__C.RANK * len(self.__C.DEVICE_IDS) for x in self.__C.DEVICE_IDS]
            device_pairs = zip(rank0_devices, self.__C.DEVICE_IDS)
            map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
            state_dict = torch.load(
                self.__C.CKPT_FILE_PATH,
                map_location=map_location)['state_dict']

            net = Net_Search(self.__C, init_dict)
            net.to(self.__C.DEVICE_IDS[0])
            net = DDP(net, device_ids=self.__C.DEVICE_IDS)
            net.load_state_dict(state_dict)

        net.eval()
        rest_data_num = eval_loader.sampler.rest_data_num
        ans_ix_list = []

        eval_loader.sampler.set_shuffle(False)
        with torch.no_grad():
            MixedOp.MODE = None
            net.module.set_chosen_op_active()
            net.module.unused_modules_off()

            cap_ix_iter_list, rel_cap_iter_list = eval_loader.dataset.get_all_caps()
            frcn_feat_iter_list, bbox_feat_iter_list, rel_img_iter_list = eval_loader.dataset.get_all_imgs()

            bs_x = self.__C.EVAL_BATCH_SIZE
            total_size_x = cap_ix_iter_list.size(0)
            col_x = math.ceil(total_size_x / bs_x)
            total_end_x = total_size_x

            total_size_y = frcn_feat_iter_list.size(0)
            row_y = math.ceil(total_size_y / self.__C.WORLD_SIZE)
            base_y = row_y * self.__C.RANK
            total_end_y = min(row_y * (self.__C.RANK + 1), total_size_y)

            scores_mat = torch.zeros(total_size_y, total_size_x).cuda(self.__C.RANK)
            for step_y in range(row_y):
                if step_y % 5 == 0 and self.__C.RANK == 0:
                    print('evaluate percent', step_y / row_y * 100.)

                start_y = base_y + step_y
                end_y = start_y + 1
                if end_y > total_end_y:
                    break
                frcn_feat_iter_ = frcn_feat_iter_list[start_y: end_y]
                bbox_feat_iter_ = bbox_feat_iter_list[start_y: end_y]
                rel_img_iter_ = rel_img_iter_list[start_y: end_y]

                for step_x in range(col_x):
                    start_x = step_x * bs_x
                    end_x = min((step_x + 1) * bs_x, total_end_x)
                    cap_ix_iter = cap_ix_iter_list[start_x: end_x]
                    rel_cap_iter = rel_cap_iter_list[start_x: end_x]
                    n_batches = cap_ix_iter.size(0)

                    frcn_feat_iter = frcn_feat_iter_.repeat(n_batches, 1, 1)
                    bbox_feat_iter = bbox_feat_iter_.repeat(n_batches, 1, 1)
                    rel_img_iter = rel_img_iter_.repeat(n_batches, 1, 1, 1)

                    eval_input_pos = (frcn_feat_iter, bbox_feat_iter, rel_img_iter, cap_ix_iter, rel_cap_iter)
                    scores_pos = net(eval_input_pos)
                    scores_mat[start_y, start_x: end_x] = scores_pos

            dist.all_reduce(scores_mat)
            net.module.unused_modules_back()

            if self.__C.RANK == 0:
                score_matrix = scores_mat.cpu().data.numpy()
                print(score_matrix.shape)

                npts = score_matrix.shape[0]
                # i2t
                stat_num = 0
                minnum_rank_image = np.array([1e7] * npts)
                for i in range(npts):
                    cur_rank = np.argsort(score_matrix[i])[::-1]
                    for index, j in enumerate(cur_rank):
                        if j in range(5 * i, 5 * i + 5):
                            stat_num += 1
                            minnum_rank_image[i] = index
                            break
                print("i2t stat num:", stat_num)

                i2t_r1 = 100.0 * len(np.where(minnum_rank_image < 1)[0]) / len(minnum_rank_image)
                i2t_r5 = 100.0 * len(np.where(minnum_rank_image < 5)[0]) / len(minnum_rank_image)
                i2t_r10 = 100.0 * len(np.where(minnum_rank_image < 10)[0]) / len(minnum_rank_image)
                i2t_medr = np.floor(np.median(minnum_rank_image)) + 1
                i2t_meanr = minnum_rank_image.mean() + 1
                print("i2t results: %.02f %.02f %.02f %.02f %.02f\n" % (i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr))

                # t2i
                stat_num = 0
                score_matrix = score_matrix.transpose()
                minnum_rank_caption = np.array([1e7] * npts * 5)
                for i in range(5 * npts):
                    img_id = i // 5
                    cur_rank = np.argsort(score_matrix[i])[::-1]
                    for index, j in enumerate(cur_rank):
                        if j == img_id:
                            stat_num += 1
                            minnum_rank_caption[i] = index
                            break

                print("t2i stat num:", stat_num)

                t2i_r1 = 100.0 * len(np.where(minnum_rank_caption < 1)[0]) / len(minnum_rank_caption)
                t2i_r5 = 100.0 * len(np.where(minnum_rank_caption < 5)[0]) / len(minnum_rank_caption)
                t2i_r10 = 100.0 * len(np.where(minnum_rank_caption < 10)[0]) / len(minnum_rank_caption)
                t2i_medr = np.floor(np.median(minnum_rank_caption)) + 1
                t2i_meanr = minnum_rank_caption.mean() + 1
                print("t2i results: %.02f %.02f %.02f %.02f %.02f\n" % (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr))

                logfile = open('./logs/log/log_' + self.__C.VERSION + '.txt', 'a+')
                logfile.write(
                    "i2t results: %.02f %.02f %.02f %.02f %.02f\n" % (i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr))
                logfile.write(
                    "t2i results: %.02f %.02f %.02f %.02f %.02f\n" % (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr))
                logfile.write("\n")
                logfile.close()

    def run(self):
        if RUN_MODE in ['train']:
            train_dataset = DataSet(self.__C, RUN_MODE)
            num_train = len(train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(self.__C.SPLIT_PORTION * num_train))

            train_sampler = SubsetDistributedSampler(train_dataset, shuffle=True,
                                                     subset_indices=indices[:split])
            search_sampler = SubsetDistributedSampler(train_dataset, shuffle=True,
                                                      subset_indices=indices[split:num_train])

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.__C.BATCH_SIZE,
                sampler=train_sampler,
                num_workers=self.__C.NUM_WORKERS,
                drop_last=True,
            )

            search_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.__C.BATCH_SIZE,
                sampler=search_sampler,
                num_workers=self.__C.NUM_WORKERS,
                drop_last=True,
            )

            eval_dataset = DataSet(self.__C, 'val')
            eval_sampler = SubsetDistributedSampler(eval_dataset, shuffle=False)
            eval_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=1,
                sampler=eval_sampler,
                num_workers=self.__C.NUM_WORKERS
            )

            neg_caps_dataset = DataSet_Neg(self.__C, keep='imgs')
            neg_caps_sampler = SubsetDistributedSampler(neg_caps_dataset, shuffle=False)
            neg_caps_loader = torch.utils.data.DataLoader(
                neg_caps_dataset,
                batch_size=self.__C.NEG_BATCHSIZE,
                sampler=neg_caps_sampler,
                num_workers=self.__C.NUM_WORKERS
            )

            neg_imgs_dataset = DataSet_Neg(self.__C, keep='caps')
            neg_imgs_sampler = SubsetDistributedSampler(neg_imgs_dataset, shuffle=False)
            neg_imgs_loader = torch.utils.data.DataLoader(
                neg_imgs_dataset,
                batch_size=self.__C.NEG_BATCHSIZE,
                sampler=neg_imgs_sampler,
                num_workers=self.__C.NUM_WORKERS
            )

            self.search(train_loader, search_loader, eval_loader, neg_caps_loader, neg_imgs_loader)


        elif RUN_MODE in ['val', 'test']:
            eval_dataset = DataSet(self.__C, RUN_MODE)
            eval_sampler = SubsetDistributedSampler(eval_dataset, shuffle=False)
            eval_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.__C.EVAL_BATCH_SIZE,
                sampler=eval_sampler,
                num_workers=self.__C.NUM_WORKERS
            )

            self.eval(eval_loader, valid=RUN_MODE in ['val'])

        else:
            exit(-1)



def mp_entrance(rank, world_size):
    __C = CfgSearch(rank, world_size)
    exec = Execution(__C)
    exec.run()


if __name__ == '__main__':
    mp.spawn(
        mp_entrance,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True
    )