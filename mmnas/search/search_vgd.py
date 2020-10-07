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

from mmnas.loader.load_data_vgd import DataSet
from mmnas.loader.filepath_vgd import Path
from mmnas.model.hygr_vgd import Net_Search
from mmnas.utils.optimizer import WarmupOptimizer
from mmnas.utils.sampler import SubsetDistributedSampler
from mmnas.model.mixed import MixedOp
from mmnas.utils.bbox_transform import clip_boxes, bbox_transform_inv
from mmnas.utils.bbox import bbox_overlaps


MASTER_ADDR = 'localhost'
MASTER_PORT = '1240'

GPU = '0, 1, 2, 3'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
# torch.set_num_threads(2)
WORLD_SIZE = len(GPU.split(','))
# WORLD_SIZE = 4
VERSION = 'run_vgd'

RUN_MODE = 'train'
# RUN_MODE = 'val'
# RUN_MODE = 'test'


class CfgSearch(Path):
    def __init__(self, rank, world_size):
        super(CfgSearch, self).__init__()
        self.PROGRESS = False

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
        random.seed(self.SEED)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        # Version Control
        self.VERSION = VERSION + '-search'
        self.RESUME = False
        self.CKPT_FILE_PATH = ''
        self.CKPT_EPOCH = 0

        ###
        self.DATASET = 'refcoco'
        # self.DATASET = 'refcoco+'
        # self.DATASET = 'refcocog'

        ###
        self.SPLIT = {
            'train': 'train',
            # 'train': 'train+val',
            'val': 'train',
            # 'val': 'val',
            'test': 'train',
            # 'test': 'testA',
            # 'test': 'testB',
            # 'test': 'test',
        }
        self.SPLIT_PORTION = 0.8

        ###
        self.IMGFEAT_MODE = 'vg_woref'
        # self.IMGFEAT_MODE = 'coco_mrcn'

        self.NUM_WORKERS = 4
        self.BATCH_SIZE = 64
        self.EVAL_BATCH_SIZE = self.BATCH_SIZE

        self.BBOX_FEATURE = False  #
        self.FRCNFEAT_LEN = 100
        self.FRCNFEAT_SIZE = 2048
        self.BBOXFEAT_EMB_SIZE = 2048
        self.GLOVE_FEATURE = True
        self.WORD_EMBED_SIZE = 300
        self.REL_SIZE = 64

        ###
        self.BBOX_NORM = True
        self.BBOX_NORM_MEANS = (0.0, 0.0, 0.0, 0.0)
        self.BBOX_NORM_STDS = (0.1, 0.1, 0.2, 0.2)
        self.OVERLAP_THRESHOLD = 0.5
        self.SCORES_LOSS = 'kld'
        self.LOSS_AVG = True
        self.LOSS_LAMBDA = 0.5

        # Network Params
        self.LAYERS = 1
        self.NODES = {
            'enc': 12,
            'dec': 18,
        }
        # self.CONCAT = list(range(2, 2 + self.NODES))
        self.HSIZE = 256
        # self.HBASE = 32
        self.DROPOUT_R = 0.1
        self.OPS_RESIDUAL = True
        self.OPS_NORM = True

        self.ATTFLAT_GLIMPSES = 1
        self.ATTFLAT_OUT_SIZE = self.HSIZE * 2
        self.ATTFLAT_MLP_SIZE = 512

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
            self.NET_LR_BASE = 0.0004
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
            self.MAX_EPOCH = 100

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


    def search(self, train_loader, eval_loader):
        # data_size = train_loader.sampler.total_size
        init_dict = {
            'token_size': train_loader.dataset.token_size,
            'pretrained_emb': train_loader.dataset.pretrained_emb,
        }

        net = Net_Search(self.__C, init_dict)
        net.to(self.__C.DEVICE_IDS[0])
        net = DDP(net, device_ids=self.__C.DEVICE_IDS)
        if self.__C.SCORES_LOSS == 'bce':
            scores_loss = torch.nn.BCEWithLogitsLoss(reduction=self.__C.REDUCTION)
        else:
            scores_loss = torch.nn.KLDivLoss(reduction=self.__C.REDUCTION)
        reg_loss = torch.nn.SmoothL1Loss(reduction=self.__C.REDUCTION).cuda()

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

        for epoch in range(start_epoch, self.__C.MAX_EPOCH):
            if self.__C.RANK == 0:
                logfile = open('./logs/log/log_' + self.__C.VERSION + '.txt', 'a+')
                logfile.write('nowTime: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
                logfile.close()

            train_loader.sampler.set_epoch(epoch)
            eval_loader.sampler.set_epoch(epoch)
            eval_loader.sampler.set_shuffle(True)
            net.train()

            if self.__C.NET_OPTIM == 'sgd':
                lr_scheduler.step()
            else:
                if epoch in self.__C.NET_LR_DECAY_LIST:
                    net_optim.decay(self.__C.NET_LR_DECAY_R)

            eval_iter = iter(eval_loader)
            for step, step_load in enumerate(train_loader):
                train_frcn_feat, train_bbox_feat, train_rel_img, train_query_ix, train_rel_query,\
                train_scores, train_scores_mask, train_transformed_bbox, train_bbox_mask, train_gt_bbox, train_bbox, train_img_shape = step_load
                train_scores = train_scores.to(self.__C.DEVICE_IDS[0])
                train_scores_mask = train_scores_mask.to(self.__C.DEVICE_IDS[0])
                train_transformed_bbox = train_transformed_bbox.to(self.__C.DEVICE_IDS[0])
                train_bbox_mask = train_bbox_mask.to(self.__C.DEVICE_IDS[0])
                train_input = (train_frcn_feat, train_bbox_feat, train_rel_img, train_query_ix, train_rel_query)

                if (step + 1) % 100 == 0 and self.__C.RANK == 0:
                    print(net.module.genotype())
                    print(net.module.genotype_weights())

                # network step
                MixedOp.MODE = None
                net.module.reset_binary_gates()
                net.module.unused_modules_off()
                pred_scores, pred_reg = net(train_input)
                # print(pred_scores.size(), train_scores.size(), train_scores_mask.size(), pred_reg.size(), train_transformed_bbox.size(), train_bbox_mask.size())
                if self.__C.SCORES_LOSS == 'bce':
                    loss_scores = scores_loss(pred_scores, train_scores)
                else:
                    loss_scores = scores_loss(pred_scores * train_scores_mask, train_scores * train_scores_mask)
                loss_reg = reg_loss(pred_reg * train_bbox_mask, train_transformed_bbox * train_bbox_mask)

                if self.__C.LOSS_AVG:
                    avg_scores = torch.sum(train_scores_mask.data)
                    avg_reg = torch.sum(train_bbox_mask.data)
                    if self.__C.SCORES_LOSS == 'bce':
                        loss_scores /= self.__C.BATCH_SIZE
                    else:
                        loss_scores /= avg_scores
                    loss_reg /= avg_reg
                loss = loss_scores + self.__C.LOSS_LAMBDA * loss_reg

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
                        eval_frcn_feat, eval_bbox_feat, eval_rel_img, eval_query_ix, eval_rel_query, \
                        eval_scores, eval_scores_mask, eval_transformed_bbox, eval_bbox_mask, eval_gt_bbox, eval_bbox, eval_img_shape = next(eval_iter)
                    except StopIteration:
                        eval_iter = iter(eval_loader)
                        eval_frcn_feat, eval_bbox_feat, eval_rel_img, eval_query_ix, eval_rel_query, \
                        eval_scores, eval_scores_mask, eval_transformed_bbox, eval_bbox_mask, eval_gt_bbox, eval_bbox, eval_img_shape = next(eval_iter)

                    eval_scores = eval_scores.to(self.__C.DEVICE_IDS[0])
                    eval_scores_mask = eval_scores_mask.to(self.__C.DEVICE_IDS[0])
                    eval_transformed_bbox = eval_transformed_bbox.to(self.__C.DEVICE_IDS[0])
                    eval_bbox_mask = eval_bbox_mask.to(self.__C.DEVICE_IDS[0])
                    eval_input = (eval_frcn_feat, eval_bbox_feat, eval_rel_img, eval_query_ix, eval_rel_query)

                    MixedOp.MODE = self.__C.ALPHA_BINARY_MODE
                    net.module.reset_binary_gates()
                    net.module.unused_modules_off()
                    pred_scores, pred_reg = net(eval_input)
                    if self.__C.SCORES_LOSS == 'bce':
                        loss_scores = scores_loss(pred_scores, eval_scores)
                    else:
                        loss_scores = scores_loss(pred_scores * eval_scores_mask, eval_scores * eval_scores_mask)
                    loss_reg = reg_loss(pred_reg * eval_bbox_mask, eval_transformed_bbox * eval_bbox_mask)

                    if self.__C.LOSS_AVG:
                        avg_scores = torch.sum(eval_scores_mask.data)
                        avg_reg = torch.sum(eval_bbox_mask.data)
                        if self.__C.SCORES_LOSS == 'bce':
                            loss_scores /= self.__C.BATCH_SIZE
                        else:
                            loss_scores /= avg_scores
                        loss_reg /= avg_reg
                    loss = loss_scores + self.__C.LOSS_LAMBDA * loss_reg

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
        # rest_data_num = eval_loader.sampler.rest_data_num
        eval_loader.sampler.set_shuffle(False)
        with torch.no_grad():
            # print(orin_state_dict['module.proj_reg.weight'])
            orin_reg_weight, orin_reg_bias = None, None
            if self.__C.BBOX_NORM:
                for name, params in net.named_parameters():
                    if 'proj_reg.weight' in name:
                        orin_reg_weight = copy.deepcopy(params.data)
                        params.data = params.data * torch.unsqueeze(torch.from_numpy(np.array(self.__C.BBOX_NORM_STDS)).to(self.__C.DEVICE_IDS[0]).float(), 1)
                    if 'proj_reg.bias' in name:
                        orin_reg_bias = copy.deepcopy(params.data)
                        params.data = params.data * torch.from_numpy(np.array(self.__C.BBOX_NORM_STDS)).to(self.__C.DEVICE_IDS[0]).float() + torch.from_numpy(np.array(self.__C.BBOX_NORM_MEANS)).to(self.__C.DEVICE_IDS[0]).float()

            MixedOp.MODE = None
            net.module.set_chosen_op_active()
            net.module.unused_modules_off()

            acc_num = 0
            all_num = 0
            for step, step_load in enumerate(eval_loader):
                # print(step, '|', len(eval_loader))
                eval_frcn_feat, eval_bbox_feat, eval_rel_img, eval_query_ix, eval_rel_query, \
                eval_scores, eval_scores_mask, eval_transformed_bbox, eval_bbox_mask, eval_gt_bbox, eval_bbox, eval_img_shape = step_load
                eval_input = (eval_frcn_feat, eval_bbox_feat, eval_rel_img, eval_query_ix, eval_rel_query)
                # torch.Size([64, 1, 4]) torch.Size([64, 100, 4]) torch.Size([64, 2])
                eval_gt_bbox = eval_gt_bbox.numpy()
                eval_bbox = eval_bbox.numpy()
                eval_img_shape = eval_img_shape.numpy()

                pred_scores, pred_reg = net(eval_input)  # torch.Size([64, 100]) torch.Size([64, 100, 4])
                cur_cuda_device = pred_scores.device
                pred_scores = pred_scores.cpu().data.numpy()
                pred_reg = pred_reg.cpu().data.numpy()
                # print(pred_scores.shape, pred_reg.shape)

                bbox_reg = bbox_transform_inv(eval_bbox.reshape(-1, 4), pred_reg.reshape(-1, 4)).reshape(-1, 100, 4)
                arg_pred_scores = np.argmax(pred_scores, axis=1)

                for step_ix in range(pred_scores.shape[0]):
                    cliped_bbox_reg_ix = clip_boxes(bbox_reg[step_ix], eval_img_shape[step_ix])
                    # print(cliped_bbox_reg_ix[arg_pred_cls[step_ix]].shape, refs_bbox[step_ix, 0].shape)
                    # overlaps = calc_iou(cliped_bbox_reg_ix[arg_pred_cls[step_ix]], refs_bbox[step_ix, 0])
                    overlaps = bbox_overlaps(
                        np.ascontiguousarray(cliped_bbox_reg_ix[arg_pred_scores[step_ix]][np.newaxis, :], dtype=np.float),
                        np.ascontiguousarray(eval_gt_bbox[step_ix], dtype=np.float))[:, 0]
                    # print(overlaps, cliped_bbox_reg_ix[arg_pred_cls[step_ix]], refs_bbox[step_ix, 0])

                    all_num += 1
                    if overlaps >= self.__C.OVERLAP_THRESHOLD:
                        acc_num += 1

            net.module.unused_modules_back()

            if self.__C.BBOX_NORM:
                for name, params in net.named_parameters():
                 if 'proj_reg.weight' in name:
                     params.data = orin_reg_weight
                 if 'proj_reg.bias' in name:
                     params.data = orin_reg_bias

            # print(acc_num, 283 + 271 + 263+ 281)
            acc_num = torch.tensor([acc_num]).to(cur_cuda_device)
            torch.distributed.all_reduce(acc_num)
            acc_num = acc_num.item()
            # print(acc_num)

            all_num = torch.tensor([all_num]).to(cur_cuda_device)
            torch.distributed.all_reduce(all_num)
            all_num = all_num.item()
            # print(all_num)

            accuracy = acc_num / float(all_num) * 100.

            if self.__C.RANK == 0:
                print('accuracy = ' + str(accuracy) + ' %')
                logfile = open('./logs/log/log_' + self.__C.VERSION + '.txt', 'a+')
                logfile.write("Overall Accuracy is: %.02f\n\n" % (accuracy))
                logfile.close()


    def run(self):
        if RUN_MODE in ['train']:
            train_dataset = DataSet(self.__C, RUN_MODE)
            num_train = len(train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(self.__C.SPLIT_PORTION * num_train))

            train_sampler = SubsetDistributedSampler(train_dataset, shuffle=True,
                                                     subset_indices=indices[:split])
            eval_sampler = SubsetDistributedSampler(train_dataset, shuffle=True,
                                                    subset_indices=indices[split:num_train])

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.__C.BATCH_SIZE,
                sampler=train_sampler,
                num_workers=self.__C.NUM_WORKERS,
                drop_last=True,
            )

            eval_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.__C.BATCH_SIZE,
                sampler=eval_sampler,
                num_workers=self.__C.NUM_WORKERS,
                drop_last=True,
            )

            self.search(train_loader, eval_loader)

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