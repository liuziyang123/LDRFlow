import numpy as np
from PIL import Image
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from torchvision.transforms import ColorJitter


class MultiImageFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        # self.eraser_aug_prob = 0.5
        self.eraser_aug_prob = 0.3

    def eraser_transform(self, img1, img2, img1_raw, img2_raw, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            mean_color_raw = np.mean(img2_raw.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
                img2_raw[y0:y0 + dy, x0:x0 + dx, :] = mean_color_raw

        return img1, img2, img1_raw, img2_raw

    def spatial_transform(self, img1, img2, img1_raw, img2_raw):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img1_raw = cv2.resize(img1_raw, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2_raw = cv2.resize(img2_raw, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                img1_raw = img1_raw[:, ::-1]
                img2_raw = img2_raw[:, ::-1]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                img1_raw = img1_raw[::-1, :]
                img2_raw = img2_raw[::-1, :]

        if img1.shape[0] == self.crop_size[0]:
            x0 = 0
            y0 = 0
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img1_raw = img1_raw[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2_raw = img2_raw[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, img1_raw, img2_raw

    def __call__(self, img1, img2, img1_raw, img2_raw):

        # occlusion
        img1, img2, img1_raw, img2_raw = self.eraser_transform(img1, img2, img1_raw, img2_raw)

        img1, img2, img1_raw, img2_raw = self.spatial_transform(img1, img2, img1_raw, img2_raw)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        img1_raw = np.ascontiguousarray(img1_raw)
        img2_raw = np.ascontiguousarray(img2_raw)

        return img1, img2, img1_raw, img2_raw