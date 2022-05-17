# from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss_train, model_loss_test, model_loss_train_self, ContrastiveCorrelationLoss
from models.submoduleEDNet import resample2d
from utils import *
from torch.utils.data import DataLoader
import gc
# from apex import amp
import cv2

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser(
    description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
parser.add_argument('--model', default='attnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/data/sceneflow/", help='data path')
parser.add_argument('--trainlist', default='./filenames/train_scene_flow.txt', help='training list')
parser.add_argument('--testlist', default='./filenames/sceneflow_test.txt', help='testing list')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=16, help='testing batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--lrepochs', default="20,32,40,44,48:2", type=str,
                    help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default='./checkpoints/model_sceneflow.ckpt',
                    help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--only_train_seg', action="store_true", help='only train seg head')
parser.add_argument('--refine_mode', action="store_true", help='use refine')
parser.add_argument('--self_supervised', action="store_true", help="use self supervised")
# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=8, drop_last=False)

# model, optimizer, loss
if args.only_train_seg:
    model = __models__[args.model](maxdisp=args.maxdisp, only_train_seg=True)
else:
    model = __models__[args.model](maxdisp=args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=0.9)

if args.self_supervised:
    lossfunction = model_loss_train_self
else:
    lossfunction = model_loss_train

if args.only_train_seg or args.refine_mode:
    contrastive_corr_loss_fn = ContrastiveCorrelationLoss()
# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt != 'none':
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
print("start at epoch {}".format(start_epoch))


def train():
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
           # if batch_idx == 20: break
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            if args.only_train_seg:
                loss = train_seg(sample)
            else:
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs = train_sample(sample, compute_metrics=do_summary)
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    print('current batch information ', scalar_outputs)
                del scalar_outputs

            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        gc.collect()

        # # testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            #if batch_idx==10:break
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            # do_summary = global_step % args.summary_freq == 0
            if args.only_train_seg:
                loss, scalar_outputs = test_seg(sample)
                avg_test_scalars.update(scalar_outputs)
            else:
                do_summary = global_step % 1 == 0
                loss, scalar_outputs = test_sample(sample)
                if do_summary:
                    save_scalars(logger, 'test', scalar_outputs, global_step)

                avg_test_scalars.update(scalar_outputs)

                print('Epoch {}/{}, Iter {}/{}, test EPE = {}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                    batch_idx,
                                                                                    len(TestImgLoader),
                                                                                    scalar_outputs["EPE"],
                                                                                    time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)

        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            # id_epoch = (epoch_idx + 1) % 100
            if args.only_train_seg:
                torch.save(checkpoint_data, "{}/checkpoint_{:0>3}_segloss_{:.3f}.ckpt".format(args.logdir, epoch_idx,
                                                                                              avg_test_scalars[
                                                                                                  "loss"]))
            else:
                torch.save(checkpoint_data, "{}/checkpoint_{:0>3}_epe_{:.3f}.ckpt".format(args.logdir, epoch_idx,
                                                                                          avg_test_scalars["EPE"][-1]))
        gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    optimizer.zero_grad()
    loss = 0
    if args.refine_mode:
        disp_ests, seg_res = model(imgL, imgR, refine_mode=True)
        fea, fea_pos, code, code_pos = seg_res
        seg_loss = 0
        for i in range(3):
            (
                pos_intra_loss, pos_intra_cd,
                pos_inter_loss, pos_inter_cd,
                neg_inter_loss, neg_inter_cd,
            ) = contrastive_corr_loss_fn(fea[i], fea_pos[i], None, None, code[i], code_pos[i])
            neg_inter_loss = neg_inter_loss.mean()
            pos_intra_loss = pos_intra_loss.mean()
            pos_inter_loss = pos_inter_loss.mean()
            seg_loss += (0.25 * pos_inter_loss +
                         0.67 * pos_intra_loss +
                         0.63 * neg_inter_loss) * 1.0
        loss += seg_loss
    else:
        disp_ests = model(imgL, imgR, refine_mode=False)
    if args.self_supervised:
        occmask = compute_occmask(model, imgL, imgR, disp_ests)
        loss += lossfunction(disp_ests, imgL, imgR, refine_mode=args.refine_mode, occ_masks=occmask)
    else:
        loss += lossfunction(disp_ests,args.maxdisp, refine_mode=args.refine_mode, disp_gt=disp_gt)
    scalar_outputs = {"loss": loss}
    loss.backward()
    optimizer.step()
    if compute_metrics:
        with torch.no_grad():
            if args.refine_mode:
                disp_ests = [disp_ests[-4], disp_ests[-1]]  # origin pixels
                scalar_outputs['seg_loss']=seg_loss
                scalar_outputs['disp_loss']=loss-seg_loss
            else:
                disp_ests = [disp_ests[-1]]
                scalar_outputs['disp_loss']=loss
            mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs)


def train_seg(sample):
    model.train()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    optimizer.zero_grad()
    fea, fea_pos, code, code_pos = model(imgL, imgR)
    loss = 0
    for i in range(3):
        (
            pos_intra_loss, pos_intra_cd,
            pos_inter_loss, pos_inter_cd,
            neg_inter_loss, neg_inter_cd,
        ) = contrastive_corr_loss_fn(fea[i], fea_pos[i], None, None, code[i], code_pos[i])
        neg_inter_loss = neg_inter_loss.mean()
        pos_intra_loss = pos_intra_loss.mean()
        pos_inter_loss = pos_inter_loss.mean()
        loss += (0.25 * pos_inter_loss +
                 0.67 * pos_intra_loss +
                 0.63 * neg_inter_loss) * 1.0

    loss.backward()
    optimizer.step()

    return tensor2float(loss)


@make_nograd_func
def test_seg(sample):
    model.eval()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()

    fea, fea_pos, code, code_pos = model(imgL, imgR)
    loss = 0
    for i in range(3):
        (
            pos_intra_loss, pos_intra_cd,
            pos_inter_loss, pos_inter_cd,
            neg_inter_loss, neg_inter_cd,
        ) = contrastive_corr_loss_fn(fea[i], fea_pos[i], None, None, code[i], code_pos[i])
        neg_inter_loss = neg_inter_loss.mean()
        pos_intra_loss = pos_intra_loss.mean()
        pos_inter_loss = pos_inter_loss.mean()
        loss += (0.25 * pos_inter_loss +
                 0.67 * pos_intra_loss +
                 0.63 * neg_inter_loss) * 1.0
    scalar_outputs = {"loss": loss}

    return tensor2float(loss), tensor2float(scalar_outputs)


@make_nograd_func
def compute_occmask(model, imgL, imgR, disp_ests):
    occ_masks = []
    imgL_rev = imgL[:, :, :, torch.arange(imgL.size(3) - 1, -1, -1)]
    imgR_rev = imgR[:, :, :, torch.arange(imgR.size(3) - 1, -1, -1)]
    if args.refine_mode:
        disp_right,_ = model(imgR_rev, imgL_rev, refine_mode=True)
    else:
        disp_right=model(imgR_rev, imgL_rev, refine_mode=False)
    disp_right = [i[:, :, torch.arange(i.size(2) - 1, -1, -1)] for i in disp_right]
    for i in range(len(disp_right)):
        disp_rec = resample2d(-disp_right[i], disp_ests[i])
        occ = ((disp_rec + disp_ests[i]) > 0.01 * (torch.abs(disp_rec) + torch.abs(disp_ests[i])) + 0.5) | (
                disp_rec == 0)  # from occlusion aware
        occ = ~occ
        occ_masks.append(occ.unsqueeze(1).repeat(1, 3, 1, 1))
    return occ_masks


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    disp_ests = model(imgL, imgR, refine_mode=args.refine_mode)

    loss = model_loss_test(disp_ests, disp_gt, mask, args.refine_mode)
    scalar_outputs = {"loss": loss, "EPE": [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests],
                      "D1": [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests],
                      "Thres1": [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests],
                      "Thres2": [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests],
                      "Thres3": [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]}

    return tensor2float(loss), tensor2float(scalar_outputs)


if __name__ == '__main__':
    train()
