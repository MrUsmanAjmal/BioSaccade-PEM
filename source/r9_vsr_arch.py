"""BasicVSR++ (Chan et al., CVPR 2022) inference architecture.

Parameter names follow the official OpenMMLab (mmediting/mmagic) generator so the
official checkpoints load with strict key checking. Modulated deformable
convolution uses torchvision.ops.deform_conv2d, whose offset/mask layout matches
the mmcv DCNv2 kernel. Loading is followed by a behavioural sanity gate in the
runner (the network must beat bicubic upsampling on held-out clean frames), so an
incorrect port cannot silently enter the comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
    n, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=x.device, dtype=x.dtype),
                                    torch.arange(0, w, device=x.device, dtype=x.dtype), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2)
    grid_flow = grid + flow
    gx = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(x, torch.stack((gx, gy), dim=3), mode=interpolation,
                         padding_mode=padding_mode, align_corners=align_corners)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        main = [nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True), nn.LeakyReLU(0.1, inplace=True)]
        main.append(nn.Sequential(*[ResidualBlockNoBN(out_channels) for _ in range(num_blocks)]))
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_conv = nn.Conv2d(in_channels, out_channels * scale_factor * scale_factor,
                                       upsample_kernel, padding=(upsample_kernel - 1) // 2)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale_factor)


class _ConvModule(nn.Module):
    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 7, 1, 3, bias=True)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x) if self.act else x


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(_ConvModule(8, 32), _ConvModule(32, 64), _ConvModule(64, 32),
                                          _ConvModule(32, 16), _ConvModule(16, 2, act=False))

    def forward(self, x):
        return self.basic_module(x)


class SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.size()
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], 2, 2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], 2, 2, count_include_pad=False))
        ref, supp = ref[::-1], supp[::-1]
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            flow_up = flow if level == 0 else F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            flow = flow_up + self.basic_module[level](torch.cat([
                ref[level], flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode='border'), flow_up], 1))
        return flow

    def forward(self, ref, supp):
        h, w = ref.shape[2:4]
        w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
        h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode='bilinear', align_corners=False)
        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w), mode='bilinear', align_corners=False)
        flow[:, 0] *= float(w) / float(w_up)
        flow[:, 1] *= float(h) / float(h_up)
        return flow


class SecondOrderDeformableAlignment(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, deform_groups=16, max_residue_magnitude=10):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * out_channels + 4, out_channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, 27 * deform_groups, 3, 1, 1))

    def forward(self, x, extra_feat, flow_1, flow_2):
        out = self.conv_offset(torch.cat([extra_feat, flow_1, flow_2], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        return deform_conv2d(x, offset, self.weight, self.bias, stride=1, padding=self.padding,
                             dilation=1, mask=torch.sigmoid(mask))


class BasicVSRPlusPlusNet(nn.Module):
    def __init__(self, mid_channels=64, num_blocks=7, max_residue_magnitude=10):
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet()
        self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        for i, module in enumerate(['backward_1', 'forward_1', 'backward_2', 'forward_2']):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels, mid_channels, 3, padding=1, deform_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ResidualBlocksWithInputConv((2 + i) * mid_channels, mid_channels, num_blocks)
        self.reconstruction = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lqs):
        n, t, c, h, w = lqs.size()
        a = lqs[:, :-1].reshape(-1, c, h, w)
        b = lqs[:, 1:].reshape(-1, c, h, w)
        backward = self.spynet(a, b).view(n, t - 1, 2, h, w)
        forward = self.spynet(b, a).view(n, t - 1, 2, h, w)
        return forward, backward

    def propagate(self, feats, flows, module_name):
        n, t, _, h, w = flows.size()
        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]
        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i]]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1]]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop]
            feat_prop = feat_prop + self.backbone[module_name](torch.cat(feat, dim=1))
            feats[module_name].append(feat_prop)
        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]
        return feats

    def upsample(self, lqs, feats):
        outputs = []
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]
        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = self.reconstruction(torch.cat(hr, dim=1))
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            outputs.append(hr + self.img_upsample(lqs[:, i]))
        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        n, t, c, h, w = lqs.size()
        if t < 2:
            raise ValueError('BasicVSR++ needs at least two frames')
        feats = {}
        f = self.feat_extract(lqs.view(-1, c, h, w)).view(n, t, -1, h, w)
        feats['spatial'] = [f[:, i] for i in range(t)]
        flows_forward, flows_backward = self.compute_flow(lqs)
        for it in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{it}'
                feats[module] = []
                flows = flows_backward if direction == 'backward' else flows_forward
                feats = self.propagate(feats, flows, module)
        return self.upsample(lqs, feats)
