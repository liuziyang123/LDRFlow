import torch
import cv2
from scipy.io import savemat
import os
from time import time


def PA_function(map, input):
    B, C, H, W = input.shape
    zeros = torch.zeros(C,1).float().cuda(map.device)
    map = torch.cat((map, zeros), 1)
    index_lower = input.floor()
    im = input.view(C, -1).squeeze()
    for c in range(input.shape[1]):
        index = index_lower[0, c, ...].view(1, -1).squeeze()
        im[c, :] = (im[c, :] - index.float()) * \
                   (map[c, ...].index_select(0, index.long()+1) - map[c, ...].index_select(0, index.long())) + \
                   map[c, ...].index_select(0, index.long())

    return im.reshape(C, H, W).unsqueeze(0)


def create_ha_table_sub(src_his, tag_his):

    src_his = src_his.float()
    tag_his = tag_his.float()

    table = torch.zeros(256).cuda(src_his.device)
    His_End = 0
    His3 = tag_his

    for i in range(256):
        if src_his[i] == 0:
            table[i] = -1
        else:
            Pix_Sum = src_his[i]
            Vua_Sum = 0
            flag = 1
            j = His_End
            while flag == 1:
                if His3[j]<Pix_Sum:
                    Pix_Sum = Pix_Sum - His3[j]
                    Vua_Sum = Vua_Sum + His3[j] * j
                    j += 1
                else:
                    His_End = j
                    His3[j] = His3[j] - Pix_Sum
                    Vua_Sum = Vua_Sum + Pix_Sum * j
                    table[i] = (Vua_Sum / src_his[i]).round()
                    flag = 0

    return table


def create_ha_table_final(src_img_, tag_img_):

    src_img = (src_img_*255).round()
    tag_img = (tag_img_*255).round()
    ha_table = list([])

    for i in range(3):
        src_img_sub = src_img[:, i, ...]
        tag_img_sub = tag_img[:, i, ...]
        src_his_sub = torch.histc(src_img_sub, bins=256, min=0, max=255)
        tag_his_sub = torch.histc(tag_img_sub, bins=256, min=0, max=255)
        ha_table_c = create_ha_table_sub(src_his_sub, tag_his_sub)
        ha_table.append(ha_table_c.unsqueeze(0))

    ha_table = torch.cat(ha_table, 0)

    return ha_table


def imf_mapping(src_img, tag_img, only_table=False):

    if only_table == False:
        output = list([])
        table_l = list([])
        for i in range(src_img.shape[0]):
            table = create_ha_table_final(src_img[i:i + 1, ...], tag_img[i:i + 1, ...])
            table_l.append(table.unsqueeze(0))
            test_img = PA_function(table, (src_img[i:i + 1, ...] * 255).round())
            output.append(test_img)
        output = torch.cat(output, 0)
        table_l = torch.cat(table_l, 0)

        return output / 255., table_l
    else:
        table_l = list([])
        for i in range(src_img.shape[0]):
            table = create_ha_table_final(src_img[i:i + 1, ...], tag_img[i:i + 1, ...])
            table_l.append(table.unsqueeze(0))
        table_l = torch.cat(table_l, 0)

        return table_l


if __name__ == '__main__':

    src_img = cv2.imread('./262A2944.png')
    tag_img = cv2.imread('./262A2945.png')

    src_img = cv2.resize(src_img, (750, 500))
    tag_img = cv2.resize(tag_img, (750, 500))

    src_img = src_img[:, :, [2, 1, 0]] / 255.
    tag_img = tag_img[:, :, [2, 1, 0]] / 255.

    src_img = torch.from_numpy(src_img).permute(2, 0, 1).unsqueeze(0).cuda().float()
    tag_img = torch.from_numpy(tag_img).permute(2, 0, 1).unsqueeze(0).cuda().float()

    start_time = time()
    table = create_ha_table_final(src_img, tag_img)
    test_img = PA_function(table, (src_img*255).round())

    img = test_img.squeeze().permute(1, 2, 0).cpu().numpy()
    img = img[:, :, [2, 1, 0]]
    cv2.imwrite('./output.png', img)