from update import BasicUpdateBlock
from extractor import BasicEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


class DocScanner(nn.Module):
    def __init__(self):
        super(DocScanner, self).__init__()

        self.hidden_dim = hdim = 160
        self.context_dim = 160

        self.fnet = BasicEncoder(output_dim=320, norm_fn='instance')
        self.update_block = BasicUpdateBlock(hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coodslar = coords_grid(N, H, W).to(img.device)
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)
        
        return coodslar, coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, iters=12, flow_init=None, test_mode=False):
        image1 = image1.contiguous()

        fmap1 = self.fnet(image1)

        warpfea = fmap1

        net, inp = torch.split(fmap1, [160, 160], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coodslar, coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            flow = coords1 - coords0

            net, up_mask, delta_flow = self.update_block(net, inp, warpfea, flow)

            coords1 = coords1 + delta_flow
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            bm_up = coodslar + flow_up

            warpfea = bilinear_sampler(fmap1, coords1.permute(0, 2, 3, 1))
            flow_predictions.append(bm_up)

        if test_mode:
            return bm_up

        return flow_predictions