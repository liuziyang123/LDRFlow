# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data

import os
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import MultiImageFlowAugmentor
from scipy.io import loadmat

DTYPE = 'tif'
MAX_IMAGE = 65535.


class EvaluateUnsupervisedFlowDataset(data.Dataset):
    def __init__(self, aug_params=None, dataset_dir=None):
        self.augmentor = MultiImageFlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.image_list = []
        self.extra_info = []
        self.dataset_dir = dataset_dir

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        raw_name1 = self.image_list[index][0]
        raw_name2 = self.image_list[index][1]
        scene = raw_name1.split('/')[-2]
        raw_name1 = raw_name1.split('/')[-1].split('.')[0]
        raw_name2 = raw_name2.split('/')[-1].split('.')[0]
        if os.path.exists(osp.join(osp.join(self.dataset_dir, scene, '%s-%s.%s'%(raw_name1, raw_name2, DTYPE)))):
            filename1 = osp.join(self.dataset_dir, scene, '%s-%s.%s'%(raw_name1, raw_name2, DTYPE))
        else:
            filename1 = osp.join(self.dataset_dir, scene, '%s.%s' % (raw_name1, DTYPE))
        if os.path.exists(osp.join(osp.join(self.dataset_dir, scene, '%s-%s.%s'%(raw_name2, raw_name1, DTYPE)))):
            filename2 = osp.join(self.dataset_dir, scene, '%s-%s.%s'%(raw_name2, raw_name1, DTYPE))
        else:
            filename2 = osp.join(self.dataset_dir, scene, '%s.%s' % (raw_name2, DTYPE))

        img1 = frame_utils.read_gen(filename1)
        img2 = frame_utils.read_gen(filename2)

        img1 = np.array(img1).astype(np.uint16)
        img2 = np.array(img2).astype(np.uint16)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        img1 = torch.from_numpy(img1.astype(np.float)).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2.astype(np.float)).permute(2, 0, 1).float()

        return img1, img2, filename1, filename2, self.extra_info[index]

    def __rmul__(self, v):
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class IMFUnsupervisedFlowDataset(data.Dataset):
    def __init__(self, aug_params=None, split=None, root=None):

        self.augmentor = MultiImageFlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.image_list = []
        self.extra_info = []

        self.raw_path = (osp.join(root, split)).replace('IMF_short2long', 'Raw')
        self.table_path = (osp.join(root, split)).replace('IMF_short2long', 'IMF_table')

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        # read corresponding raw 8bit image
        filename1_l = self.image_list[index][0].split('/')
        filename2_l = self.image_list[index][1].split('/')
        filename1 = filename1_l[-1].split('-')[0].split('.')[0]
        filename2 = filename2_l[-1].split('-')[0].split('.')[0]
        img1_raw = frame_utils.read_gen(self.raw_path+filename1_l[-2]+'/'+filename1+'.tif')
        img2_raw = frame_utils.read_gen(self.raw_path+filename1_l[-2]+'/'+filename2+'.tif')

        # read the histogram table
        table12 = loadmat(self.table_path+filename1_l[-2]+'/'+filename1+'-'+filename2+'.mat')['T']
        table21 = loadmat(self.table_path+filename1_l[-2]+'/'+filename2+'-'+filename1+'.mat')['T']

        img1 = np.array(img1).astype(np.uint16)
        img2 = np.array(img2).astype(np.uint16)
        img1_raw = np.array(img1_raw).astype(np.uint16)
        img2_raw = np.array(img2_raw).astype(np.uint16)

        H, W, _ = img1.shape
        if H > W:
            img1 = np.transpose(img1, (1, 0, 2))
            img2 = np.transpose(img2, (1, 0, 2))
            img1_raw = np.transpose(img1_raw, (1, 0, 2))
            img2_raw = np.transpose(img2_raw, (1, 0, 2))

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        img1, img2, img1_raw, img2_raw = self.augmentor(img1, img2, img1_raw, img2_raw)

        img1 = torch.from_numpy(img1.astype(np.float)).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2.astype(np.float)).permute(2, 0, 1).float()
        img1_raw = torch.from_numpy(img1_raw.astype(np.float)).permute(2, 0, 1).float()
        img2_raw = torch.from_numpy(img2_raw.astype(np.float)).permute(2, 0, 1).float()
        table12 = torch.from_numpy(table12.astype(np.float)).float()
        table21 = torch.from_numpy(table21.astype(np.float)).float()

        return img1, img2, img1_raw, img2_raw, table21.squeeze(), table12.squeeze()

    def __rmul__(self, v):
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class evaluate_hdr_IMF(EvaluateUnsupervisedFlowDataset):
    def __init__(self, aug_params=None, split='Test', root='./data/IMF_short2long', dstype='Sen'):
        super(evaluate_hdr_IMF, self).__init__(aug_params, dataset_dir=osp.join(root, split, dstype))
        raw_dir = root.replace('IMF_short2long', 'Raw')
        raw_root = osp.join(raw_dir, split, dstype)

        if split == 'Test':
            self.is_test = True

        for scene in os.listdir(raw_root):
            image_list = sorted(glob(osp.join(raw_root, scene, '*.%s'%(DTYPE))))
            target_index = len(image_list) // 2
            for i in range(len(image_list)):
                if i != target_index:
                    src_file = image_list[i]
                    tag_file = image_list[target_index]
                    if os.path.exists(src_file) and os.path.exists(tag_file):
                        self.image_list += [[image_list[i], image_list[target_index]]]
                self.extra_info += [(scene, i)]


class hdr_IMF(IMFUnsupervisedFlowDataset):
    def __init__(self, aug_params=None, split='Training', root='./data/IMF_short2long'):
        super(hdr_IMF, self).__init__(aug_params, split, root)
        image_root = osp.join(root, split)

        if split == 'Test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.tif')))
            for i in range(len(image_list)):
                if '-' in image_list[i]:
                    filename = image_list[i].split('-')[-1]
                    file_name_raw = image_list[i].split('/')[-1]
                    self.image_list += [[image_list[i], image_list[i].replace(file_name_raw, filename)]]


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'ablation':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        train_dataset = hdr_IMF(aug_params, split='Training')

    elif args.stage == 'sota':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        train_dataset_1 = hdr_IMF(aug_params, split='Training')
        train_dataset_2 = hdr_IMF(aug_params, split='Training_ICCV')
        train_dataset = 40 * train_dataset_1 + train_dataset_2

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
        pin_memory=False, shuffle=True, num_workers=8, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

