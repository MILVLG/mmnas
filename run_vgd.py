import math, os, json, torch, datetime, random, copy, shutil, torchvision, tqdm
import argparse, yaml
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
from mmnas.model.full_vgd import Net_Full
from mmnas.utils.optimizer import WarmupOptimizer
from mmnas.utils.sampler import SubsetDistributedSampler
from mmnas.utils.bbox_transform import clip_boxes, bbox_transform_inv
from mmnas.utils.bbox import bbox_overlaps

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='MmNas Args')

    parser.add_argument('--RUN', dest='RUN_MODE', default='train',
                      choices=['train', 'val', 'test'],
                      help='{train, val, test}',
                      type=str)

    parser.add_argument('--DATASET', dest='DATASET', default='refcoco',
                      choices=['refcoco', 'refcoco+', 'refcocog'],
                      help='{refcoco, refcoco+, refcocog}',
                      type=str)

    parser.add_argument('--FEAT', dest='FEAT', default='vg_woref',
                      choices=['vg_woref', 'coco_mrcn'],
                      help='{vg_woref, coco_mrcn}',
                      type=str)

    parser.add_argument('--SPLIT', dest='TRAIN_SPLIT', default='train',
                      choices=['train', 'train+val'],
                      help="set training split, "
                           "eg.'train', 'train+val+vg'"
                           "set 'train' can trigger the "
                           "eval after every epoch",
                      type=str)

    parser.add_argument('--BS', dest='BATCH_SIZE', default=64,
                      help='batch size during training',
                      type=int)

    parser.add_argument('--NW', dest='NUM_WORKERS', default=4,
                      help='fix random seed',
                      type=int)

    parser.add_argument('--ARCH_PATH', dest='ARCH_PATH', default='./arch/run_vgd.json',
                      help='version control',
                      type=str)

    parser.add_argument('--GENO_EPOCH', dest='GENO_EPOCH', default=0,
                      help='version control',
                      type=int)

    parser.add_argument('--GPU', dest='GPU', default='0',
                      help="gpu select, eg.'0, 1, 2'",
                      type=str)

    parser.add_argument('--SEED', dest='SEED', default=None,
                      help='fix random seed',
                      type=int)

    parser.add_argument('--VERSION', dest='VERSION', default='run_vgd',
                      help='version control',
                      type=str)

    parser.add_argument('--RESUME', dest='RESUME', default=False,
                      help='resume training',
                      action='store_true')

    parser.add_argument('--CKPT_PATH', dest='CKPT_FILE_PATH',
                      help='load checkpoint path',
                      type=str)

    args = parser.parse_args()
    return args


class Cfg(Path):
    def __init__(self, rank, world_size, args):
        super(Cfg, self).__init__()

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = 12340 if world_size > 1 else str(random.randint(10000, 20000))
        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # self.DEBUG = True
        self.DEBUG = False

        # Set Devices
        self.WORLD_SIZE = world_size
        self.RANK = rank
        self.N_GPU = torch.cuda.device_count() // self.WORLD_SIZE
        self.DEVICE_IDS = list(range(self.RANK * self.N_GPU, (self.RANK + 1) * self.N_GPU))

        # Set Seed For CPU And GPUs
        if args.SEED is None:
            self.SEED = random.randint(0, 9999)
        else:
            self.SEED = args.SEED
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)
        np.random.seed(self.SEED)
        random.seed(self.SEED)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        # Version Control
        self.VERSION = args.VERSION + '-full'
        self.RESUME = args.RESUME
        self.CKPT_FILE_PATH = args.CKPT_FILE_PATH

        self.DATASET = args.DATASET
        self.IMGFEAT_MODE = args.FEAT
        self.SPLIT = {
            'train': args.TRAIN_SPLIT,
            'val': 'val',
            'test': 'testA',
            # 'test': 'testB',
        }
        self.EVAL_EVERY_EPOCH = True

        self.TEST_SAVE_PRED = False
        if self.SPLIT['val'] in self.SPLIT['train'].split('+') or args.RUN_MODE not in ['train']:
            self.EVAL_EVERY_EPOCH = False
        print('Eval after every epoch: ', self.EVAL_EVERY_EPOCH)

        self.NUM_WORKERS = args.NUM_WORKERS
        self.BATCH_SIZE = args.BATCH_SIZE
        self.EVAL_BATCH_SIZE = self.BATCH_SIZE

        self.BBOX_FEATURE = False
        self.FRCNFEAT_LEN = 100
        self.FRCNFEAT_SIZE = 2048
        self.BBOXFEAT_EMB_SIZE = 2048
        self.GLOVE_FEATURE = True
        self.WORD_EMBED_SIZE = 300
        self.REL_SIZE = 64

        self.BBOX_NORM = True
        self.BBOX_NORM_MEANS = (0.0, 0.0, 0.0, 0.0)
        self.BBOX_NORM_STDS = (0.1, 0.1, 0.2, 0.2)
        self.OVERLAP_THRESHOLD = 0.5
        self.SCORES_LOSS = 'kld'
        self.LOSS_AVG = True
        self.LOSS_LAMBDA = 0.5

        # Network Params
        self.LAYERS = 1
        self.HSIZE = 512
        # self.HBASE = 64
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
            self.NET_LR_BASE = 0.01
            self.NET_LR_MIN = 0.004
            self.NET_MOMENTUM = 0.9
            # self.NET_WEIGHT_DECAY = 3e-5
            self.NET_WEIGHT_DECAY = 0
            # self.NET_GRAD_CLIP = 1.  # GRAD_CLIP = -1: means not use grad_norm_clip
            self.NET_GRAD_CLIP = -1  # GRAD_CLIP = -1: means not use grad_norm_clip
            self.MAX_EPOCH = 20

        else:
            self.NET_OPTIM_WARMUP = True
            self.NET_LR_BASE = 0.00014
            # self.NET_WEIGHT_DECAY = 3e-5
            self.NET_WEIGHT_DECAY = 0
            self.NET_GRAD_CLIP = 1.  # GRAD_CLIP = -1: means not use grad_norm_clip
            # self.NET_GRAD_CLIP = -1  # GRAD_CLIP = -1: means not use grad_norm_clip
            self.NET_LR_DECAY_R = 0.2
            self.NET_LR_DECAY_LIST = [10, 12]
            self.OPT_BETAS = (0.9, 0.98)
            self.OPT_EPS = 1e-9
            self.MAX_EPOCH = 13

        self.GENOTYPE = json.load(open(args.ARCH_PATH, 'r+'))['epoch' + str(args.GENO_EPOCH)]
        self.REDUMP_EVAL = False

        if self.RANK == 0:
            print('Use the GENOTYPE PATH:', args.ARCH_PATH)
            print('Use the GENOTYPE EPOCH:', args.GENO_EPOCH)
            print(self.GENOTYPE)


class Execution:
    def __init__(self, __C):
        self.__C = __C

    def get_optim(self, net, search=False, epoch_steps=None):
        net_optim = None
        alpha_optim = None

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

        return net_optim, alpha_optim


    def train(self, train_loader, eval_loader):
        # data_size = train_loader.sampler.total_size
        init_dict = {
            'token_size': train_loader.dataset.token_size,
            'pretrained_emb': train_loader.dataset.pretrained_emb,
        }

        net = Net_Full(self.__C, init_dict)
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
            start_epoch = ckpt['epoch']
            net_optim, _ = self.get_optim(net, search=False, epoch_steps=len(train_loader))
            if self.__C.NET_OPTIM == 'sgd':
                net_optim.load_state_dict(ckpt['net_optim'])
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    net_optim, self.__C.MAX_EPOCH, last_epoch=start_epoch)
            else:
                net_optim.optimizer.load_state_dict(ckpt['net_optim'])
                net_optim.set_start_step(start_epoch * len(train_loader))

        else:
            net_optim, _ = self.get_optim(net, search=False, epoch_steps=len(train_loader))
            start_epoch = 0

            lr_scheduler = None
            if self.__C.NET_OPTIM == 'sgd':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    net_optim, self.__C.MAX_EPOCH)

        loss_sum = 0
        named_params = list(net.named_parameters())

        for epoch in range(start_epoch, self.__C.MAX_EPOCH):
            if self.__C.RANK == 0:
                logfile = open('./logs/log/log_' + self.__C.VERSION + '.txt', 'a+')
                logfile.write('nowTime: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
                logfile.close()

            train_loader.sampler.set_epoch(epoch)
            net.train()

            if self.__C.NET_OPTIM == 'sgd':
                lr_scheduler.step()
            else:
                if epoch in self.__C.NET_LR_DECAY_LIST:
                    net_optim.decay(self.__C.NET_LR_DECAY_R)

            for step, step_load in enumerate(tqdm.tqdm(train_loader)):
                train_frcn_feat, train_bbox_feat, train_rel_img, train_query_ix, train_rel_query, \
                train_scores, train_scores_mask, train_transformed_bbox, train_bbox_mask, train_gt_bbox, train_bbox, train_img_shape = step_load
                train_scores = train_scores.to(self.__C.DEVICE_IDS[0])
                train_scores_mask = train_scores_mask.to(self.__C.DEVICE_IDS[0])
                train_transformed_bbox = train_transformed_bbox.to(self.__C.DEVICE_IDS[0])
                train_bbox_mask = train_bbox_mask.to(self.__C.DEVICE_IDS[0])
                train_input = (train_frcn_feat, train_bbox_feat, train_rel_img, train_query_ix, train_rel_query)

                # network step
                net_optim.zero_grad()
                pred_scores, pred_reg = net(train_input)
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
                loss.backward()
                loss_sum += loss.item()

                # if self.__C.DEBUG and self.__C.RANK == 0:
                #     if self.__C.REDUCTION == 'sum':
                #         print(step, loss.item() / self.__C.BATCH_SIZE)
                #     else:
                #         print(step, loss.item())

                # gradient clipping
                if self.__C.NET_GRAD_CLIP > 0:
                    nn.utils.clip_grad_norm_(net.parameters(), self.__C.NET_GRAD_CLIP)
                net_optim.step()

            epoch_finish = epoch + 1

            if self.__C.RANK == 0:
                state = {
                    'state_dict': net.state_dict(),
                    'net_optim': net_optim.state_dict() if self.__C.NET_OPTIM == 'sgd' else net_optim.optimizer.state_dict(),
                    'epoch': epoch_finish,
                }
                torch.save(state, self.__C.CKPT_PATH + self.__C.VERSION + '_epoch' + str(epoch_finish) + '.pkl')

                if self.__C.NET_OPTIM == 'sgd':
                    lr_cur = lr_scheduler.get_lr()[0]
                else:
                    lr_cur = net_optim._rate

                logfile = open('./logs/log/log_' + self.__C.VERSION + '.txt', 'a+')
                # logfile.write('epoch = ' + str(epoch_finish) + '  loss = ' + str(loss_sum / data_size) + '\n' +
                #               'lr = ' + str(optim._rate) + '\n')
                if self.__C.REDUCTION == 'sum':
                    logfile.write('epoch = ' + str(epoch_finish) + '  loss = ' +
                                  str(loss_sum / len(train_loader) / self.__C.BATCH_SIZE) +
                                  '\n' + 'lr = ' + str(lr_cur) + '\n')
                else:
                    logfile.write('epoch = ' + str(epoch_finish) + '  loss = ' + str(loss_sum / len(train_loader)) +
                                  '\n' + 'lr = ' + str(lr_cur) + '\n')
                logfile.close()

            dist.barrier()

            if eval_loader is not None:
                self.eval(
                    eval_loader,
                    net=net,
                    valid=True,
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

            net = Net_Full(self.__C, init_dict)
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

            acc_num = 0
            all_num = 0
            for step, step_load in enumerate(tqdm.tqdm(eval_loader)):
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


    def run(self, args):
        if args.RUN_MODE in ['train']:
            train_dataset = DataSet(self.__C, args.RUN_MODE)
            train_sampler = SubsetDistributedSampler(train_dataset, shuffle=True)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.__C.BATCH_SIZE,
                sampler=train_sampler,
                num_workers=self.__C.NUM_WORKERS,
                drop_last=True
            )

            eval_loader = None
            if self.__C.EVAL_EVERY_EPOCH:
                eval_dataset = DataSet(self.__C, 'val')
                eval_sampler = SubsetDistributedSampler(eval_dataset, shuffle=False)
                eval_loader = torch.utils.data.DataLoader(
                    eval_dataset,
                    batch_size=self.__C.EVAL_BATCH_SIZE,
                    sampler=eval_sampler,
                    num_workers=self.__C.NUM_WORKERS
                )

            self.train(train_loader, eval_loader)


        elif args.RUN_MODE in ['val', 'test']:
            eval_dataset = DataSet(self.__C, args.RUN_MODE)
            eval_sampler = SubsetDistributedSampler(eval_dataset, shuffle=False)
            eval_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.__C.EVAL_BATCH_SIZE,
                sampler=eval_sampler,
                num_workers=self.__C.NUM_WORKERS
            )

            self.eval(eval_loader, valid=args.RUN_MODE in ['val'])

        else:
            exit(-1)


def mp_entrance(rank, world_size, args):
    __C = Cfg(rank, world_size, args)
    exec = Execution(__C)
    exec.run(args)


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    WORLD_SIZE = len(args.GPU.split(','))

    mp.spawn(
        mp_entrance,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True
    )