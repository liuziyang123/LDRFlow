import sys
sys.path.append('core')

import argparse
import os
import numpy as np
import torch

import datasets

from raft import RAFT
from utils.utils import InputPadder
from utils.flow_viz import viz_write

from tqdm import tqdm
from scipy.io import loadmat

MAX_IMAGE = 65535.


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def validate_hdr_IMF_downsample(model, experiment_name, iters=32):

    model.eval()
    results = {}

    gt_dir = './GT_flow'
    save_dir = './results/' + experiment_name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for dstype in ['SIG17_PAPER', 'SIG17_EXTRA', 'Sen', 'Tursun', 'ICCP19']:

        epe_list = []

        epe_10_list = []
        epe_10_40_list = []
        epe_40_list = []

        val_dataset = datasets.evaluate_hdr_IMF(split='Test', dstype=dstype)

        for val_id in tqdm(range(len(val_dataset))):

            image1, image2, filename1, filename2, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            name1_l = filename1.split('/')
            name2_l = filename2.split('/')

            # load gt flow computed by traditional methods
            gt_name = '%s/%s*%s*%s*%s.mat' % (gt_dir, name1_l[-3], name1_l[-2], name2_l[-1].split('.')[0], name1_l[-1].split('.')[0])
            gt_flow = loadmat(gt_name)['flow']
            gt_flow = torch.from_numpy(gt_flow).permute(2, 0, 1).unsqueeze(0)
            gt_flow = gt_flow.cuda(image1.device)
            flow_range = (gt_flow[:, 0, ...] ** 2 + gt_flow[:, 1, ...] ** 2).sqrt()
            flow_range = flow_range.unsqueeze(1)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # computes the optical flow
            flow_low, flow_pr = model(image2, image1, iters=iters, test_mode=True)
            flow_pr = padder.unpad(flow_pr)

            epe = torch.sum((flow_pr - gt_flow)**2, dim=1).sqrt()

            epe_list.append(epe.view(-1).cpu().numpy())

            # epe in different range
            epe = epe.unsqueeze(0)
            val = (flow_range < 10)
            if val.max() > 0:
                epe_10_list.append(epe[val].view(-1).cpu().numpy())
            val = (flow_range >= 10) * (flow_range <= 40)
            if val.max() > 0:
                epe_10_40_list.append(epe[val].view(-1).cpu().numpy())
            val = flow_range > 40
            if val.max() > 0:
                epe_40_list.append(epe[val].view(-1).cpu().numpy())

            viz_write(flow_pr, save_dir, dstype + '_' + filename1.split('/')[-2] + '_' + filename1.split('/')[-1])

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        print(dstype)
        print("EPE")
        print(epe)
        print("<10")
        print(np.mean(np.concatenate(epe_10_list)))
        print("10-40")
        print(np.mean(np.concatenate(epe_10_40_list)))
        print(">40")
        print(np.mean(np.concatenate(epe_40_list)))

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    print("Parameter Count: %d" % count_parameters(model))

    model.load_state_dict(torch.load(args.model, map_location='cpu'), strict=True)

    model.cuda()
    model.eval()

    with torch.no_grad():
        validate_hdr_IMF_downsample(model.module, args.model.split('/')[-1])

