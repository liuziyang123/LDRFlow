import torch
import torch.nn as nn
import torch.nn.functional as F
MAX_IMAGE = 65535.


# warp
class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid[:, [1, 0], ...] + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[-i-1] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode="border", align_corners=False)


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, th=1.0):

        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, th)


# smooth loss
def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


# omega
def omega_function(input):
    global MAX_IMAGE

    f1 = input + 1/MAX_IMAGE
    f2 = (MAX_IMAGE+1)/MAX_IMAGE - input

    inter1 = input < ((MAX_IMAGE+1)/2/MAX_IMAGE)
    inter2 = input >= ((MAX_IMAGE+1)/2/MAX_IMAGE)

    return f1*inter1 + f2*inter2


# proposed imf-based learning objective
def imf_objective(image1, image1_mapped, image1_pre, image1_pre_mapped, th=0.6):

    omega_image1 = omega_function(image1)
    omega_image1_pre = omega_function(image1_pre)

    inter1 = omega_image1_pre > omega_image1
    inter2 = omega_image1_pre <= omega_image1

    tag = image1 * inter1 + inter2 * image1_mapped
    pre = image1_pre_mapped * inter1 + inter2 * image1_pre

    loss = (tag-pre).abs()

    return torch.clamp(loss, max=th), inter1, inter2