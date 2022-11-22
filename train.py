from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from raft import RAFT

import datasets

from torch.utils.tensorboard import SummaryWriter

from self_supervised_utils import SpatialTransformer, SSIM, get_smooth_loss, imf_objective
from imf_utils import PA_function

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
SUM_FREQ = 10
SAVE_FREQ = 10000
MAX_IMAGE = 65535.


def unsupervised_sequence_loss(image1, image2, flows_12, warp, ssim, table21=None, table12=None, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flows_12)
    flow_loss = 0.0
    issim_sum = 0.0
    smooth_sum = 0.0
    imae_sum = 0.0

    # IMF mapping
    image1_mapped = list([])
    for j in range(image1.shape[0]):
        image1_mapped.append(PA_function(table12[j:j + 1, ...].squeeze(), (image1[j:j + 1, ...] * MAX_IMAGE)) / MAX_IMAGE)
    image1_mapped = torch.cat(image1_mapped, 0)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)

        # warp
        image1_pre = warp(image2, flows_12[i])
        # enforce image1_pre into 0-1
        image1_pre = torch.clamp(image1_pre, 0, 1)

        # IMF mapping
        image1_pre_mapped = list([])
        for j in range(image1.shape[0]):
            image1_pre_mapped.append(PA_function(table21[j:j+1, ...].squeeze(), (image1_pre[j:j+1, ...] * MAX_IMAGE)) / MAX_IMAGE)
        image1_pre_mapped = torch.cat(image1_pre_mapped, 0)

        imae_loss, mask1, mask2 = imf_objective(image1, image1_mapped, image1_pre, image1_pre_mapped, th=0.15)
        issim_loss = (ssim(image1, image1_pre_mapped) * mask1 + ssim(image1_mapped, image1_pre) * mask2)
        issim_loss = torch.clamp(issim_loss, max=0.15)
        smooth_loss = get_smooth_loss(flows_12[i], image1)

        imae_sum += imae_loss.mean()
        issim_sum += issim_loss.mean()
        smooth_sum += smooth_loss

        flow_loss += i_weight * imae_loss.mean() * 0.15 + \
                     i_weight * issim_loss.mean() * 0.85 + i_weight * smooth_loss * 0.01

    metrics = {
        'imae_loss': imae_sum.item(),
        'smooth_loss': smooth_sum.item(),
        'issim_loss': issim_sum.item(),
        'flow_loss': flow_loss.item()
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

        self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], self.total_steps)

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt, map_location='cpu'), strict=True)

    model.to(torch.device('cuda:' + str(args.gpus[0])))
    model.train()

    warp = SpatialTransformer(args.image_size).cuda('cuda:'+str(model.device_ids[0]))
    ssim = SSIM()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, image1_raw, image2_raw, table21_l, table12_l = [x.to(torch.device('cuda:'+str(args.gpus[0]))) for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 0.019*MAX_IMAGE)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, MAX_IMAGE)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, MAX_IMAGE)

            flows_12 = model(image1, image2, iters=args.iters)
            flows_21 = model(image2, image1, iters=args.iters)

            loss12, metrics = unsupervised_sequence_loss(image1_raw/MAX_IMAGE, image2_raw/MAX_IMAGE, flows_12, warp, ssim, table21_l, table12_l, args.gamma)
            loss21, _ = unsupervised_sequence_loss(image2_raw/MAX_IMAGE, image1_raw/MAX_IMAGE, flows_21, warp, ssim, table12_l, table21_l, args.gamma)

            loss = loss12 + loss21

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % SAVE_FREQ == SAVE_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='LDRFlow', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')

    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--num_steps', type=int, default=150000)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 704])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)