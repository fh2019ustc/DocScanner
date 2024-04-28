from model import DocScanner
from seg import U2NETP

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
import argparse

import warnings
warnings.filterwarnings('ignore')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.msk = U2NETP(3, 1)
        self.bm = DocScanner()  # 矫正

    def forward(self, x):
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x

        bm = self.bm(x, iters=12, test_mode=True)
        bm = (2 * (bm / 286.8) - 1) * 0.99

        return bm


def reload_seg_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def reload_rec_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def rec(seg_model_path, rec_model_path, distorrted_path, save_path):
    # distorted images list
    img_list = os.listdir(distorrted_path)

    # creat save path for rectified images
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # net init
    net = Net().cuda()
    # reload seg model
    reload_seg_model(net.msk, seg_model_path)
    # reload rec model
    reload_rec_model(net.bm, rec_model_path)

    net.eval()

    for img_path in img_list:
        name = img_path.split('.')[-2]  # image name
        img_path = distorrted_path + img_path  # image path

        im_ori = np.array(Image.open(img_path))[:, :, :3] / 255.
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)

        with torch.no_grad():
            bm = net(im.cuda())
            bm = bm.cpu()

            # save rectified image
            bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))  # x flow
            bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))  # y flow
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))
            lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)  # h * w * 2
            out = F.grid_sample(torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
            cv2.imwrite(save_path + name + '_rec' + '.png', (((out[0]*255).permute(1, 2, 0).numpy())[:,:,::-1]).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_model_path', default='./model_pretrained/seg.pth')
    parser.add_argument('--rec_model_path', default='./model_pretrained/DocScanner-L.pth')
    parser.add_argument('--distorrted_path', default='./distorted/')
    parser.add_argument('--rectified_path', default='./rectified/')
    opt = parser.parse_args()

    rec(seg_model_path=opt.seg_model_path,
        rec_model_path=opt.rec_model_path,
        distorrted_path=opt.distorrted_path,
        save_path=opt.rectified_path)


if __name__ == "__main__":
    main()
