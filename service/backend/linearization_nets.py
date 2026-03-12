"""PyTorch SingleHDR dequantization + linearization networks.
Copied from singlehdr-service/torch_nets.py (only the nets needed for linearization).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Dequantization-Net (U-Net)
# ============================================================================

class DequantizationNet(nn.Module):
    """U-Net encoder-decoder with skip connections for removing quantization artifacts."""

    def __init__(self):
        super().__init__()
        self.conv_in1 = nn.Conv2d(3, 16, 7, padding=3)
        self.conv_in2 = nn.Conv2d(16, 16, 7, padding=3)

        self.down1_c1 = nn.Conv2d(16, 32, 5, padding=2)
        self.down1_c2 = nn.Conv2d(32, 32, 5, padding=2)
        self.down2_c1 = nn.Conv2d(32, 64, 3, padding=1)
        self.down2_c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.down3_c1 = nn.Conv2d(64, 128, 3, padding=1)
        self.down3_c2 = nn.Conv2d(128, 128, 3, padding=1)
        self.down4_c1 = nn.Conv2d(128, 256, 3, padding=1)
        self.down4_c2 = nn.Conv2d(256, 256, 3, padding=1)

        self.up1_c1 = nn.Conv2d(256, 128, 3, padding=1)
        self.up1_c2 = nn.Conv2d(256, 128, 3, padding=1)
        self.up2_c1 = nn.Conv2d(128, 64, 3, padding=1)
        self.up2_c2 = nn.Conv2d(128, 64, 3, padding=1)
        self.up3_c1 = nn.Conv2d(64, 32, 3, padding=1)
        self.up3_c2 = nn.Conv2d(64, 32, 3, padding=1)
        self.up4_c1 = nn.Conv2d(32, 16, 3, padding=1)
        self.up4_c2 = nn.Conv2d(32, 16, 3, padding=1)

        self.conv_out = nn.Conv2d(16, 3, 3, padding=1)

    def _down(self, x, c1, c2):
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(c1(x), 0.1)
        x = F.leaky_relu(c2(x), 0.1)
        return x

    def _up(self, x, skip, c1, c2):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.leaky_relu(c1(x), 0.1)
        x = torch.cat([x, skip], dim=1)
        x = F.leaky_relu(c2(x), 0.1)
        return x

    def forward(self, inp):
        x = F.leaky_relu(self.conv_in1(inp), 0.1)
        s1 = F.leaky_relu(self.conv_in2(x), 0.1)
        s2 = self._down(s1, self.down1_c1, self.down1_c2)
        s3 = self._down(s2, self.down2_c1, self.down2_c2)
        s4 = self._down(s3, self.down3_c1, self.down3_c2)
        x = self._down(s4, self.down4_c1, self.down4_c2)
        x = self._up(x, s4, self.up1_c1, self.up1_c2)
        x = self._up(x, s3, self.up2_c1, self.up2_c2)
        x = self._up(x, s2, self.up3_c1, self.up3_c2)
        x = self._up(x, s1, self.up4_c1, self.up4_c2)
        x = torch.tanh(self.conv_out(x))
        return inp + x


# ============================================================================
# Linearization-Net (CrfFeatureNet + AEInvcrfDecodeNet)
# ============================================================================

class _ConvBN(nn.Module):
    def __init__(self, c_in, c_out, k, s, bias=True, has_bn=False, has_relu=True):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(c_out) if has_bn else None
        self.has_relu = has_relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.has_relu:
            x = F.relu(x)
        return x


class CrfFeatureNet(nn.Module):
    """ResNet50-like feature extractor for camera response function estimation."""

    def __init__(self):
        super().__init__()
        self.conv1 = _ConvBN(102, 64, 7, 2, bias=True, has_bn=True, has_relu=True)
        self.pool1 = nn.MaxPool2d(3, 2, 1)

        self.res2a_b1 = _ConvBN(64, 256, 1, 1, bias=False, has_bn=True, has_relu=False)
        self.res2a_b2a = _ConvBN(64, 64, 1, 1, bias=False, has_bn=True, has_relu=True)
        self.res2a_b2b = _ConvBN(64, 64, 3, 1, bias=False, has_bn=True, has_relu=True)
        self.res2a_b2c = _ConvBN(64, 256, 1, 1, bias=False, has_bn=True, has_relu=False)

        self.res2b_b2a = _ConvBN(256, 64, 1, 1, bias=False, has_bn=True, has_relu=True)
        self.res2b_b2b = _ConvBN(64, 64, 3, 1, bias=False, has_bn=True, has_relu=True)
        self.res2b_b2c = _ConvBN(64, 256, 1, 1, bias=False, has_bn=True, has_relu=False)

        self.res2c_b2a = _ConvBN(256, 64, 1, 1, bias=False, has_bn=True, has_relu=True)
        self.res2c_b2b = _ConvBN(64, 64, 3, 1, bias=False, has_bn=True, has_relu=True)
        self.res2c_b2c = _ConvBN(64, 256, 1, 1, bias=False, has_bn=True, has_relu=False)

        self.res3a_b1 = _ConvBN(256, 512, 1, 2, bias=False, has_bn=True, has_relu=False)
        self.res3a_b2a = _ConvBN(256, 128, 1, 2, bias=False, has_bn=True, has_relu=True)
        self.res3a_b2b = _ConvBN(128, 128, 3, 1, bias=False, has_bn=True, has_relu=True)
        self.res3a_b2c = _ConvBN(128, 512, 1, 1, bias=False, has_bn=True, has_relu=False)

        self.res3b_b2a = _ConvBN(512, 128, 1, 1, bias=False, has_bn=True, has_relu=True)
        self.res3b_b2b = _ConvBN(128, 128, 3, 1, bias=False, has_bn=True, has_relu=True)
        self.res3b_b2c = _ConvBN(128, 512, 1, 1, bias=False, has_bn=True, has_relu=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        b1 = self.res2a_b1(x)
        b2 = self.res2a_b2c(self.res2a_b2b(self.res2a_b2a(x)))
        x = F.relu(b1 + b2)

        b2 = self.res2b_b2c(self.res2b_b2b(self.res2b_b2a(x)))
        x = F.relu(x + b2)

        b2 = self.res2c_b2c(self.res2c_b2b(self.res2c_b2a(x)))
        x = F.relu(x + b2)

        b1 = self.res3a_b1(x)
        b2 = self.res3a_b2c(self.res3a_b2b(self.res3a_b2a(x)))
        x = F.relu(b1 + b2)

        b2 = self.res3b_b2c(self.res3b_b2b(self.res3b_b2a(x)))
        x = F.relu(x + b2)

        x = x.mean(dim=[2, 3])
        return x


class AEInvcrfDecodeNet(nn.Module):
    """Decodes a feature vector into a 1024-point inverse CRF using PCA basis."""

    def __init__(self, invemor_path):
        super().__init__()
        self.fc = nn.Linear(512, 11)

        g0, hinv = self._parse_invemor(invemor_path)
        self.register_buffer('g0', torch.from_numpy(g0).float())
        self.register_buffer('hinv', torch.from_numpy(hinv).float())

    @staticmethod
    def _parse_invemor(path):
        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]

        def _parse(tag):
            for i, line in enumerate(lines):
                if line == tag:
                    break
            start = i + 1
            vals = []
            for j in range(start, start + 256):
                vals += lines[j].split()
            return np.float32(vals)

        g0 = _parse('g0 =')
        hinv = np.stack([_parse('hinv(%d)=' % (i + 1)) for i in range(11)], axis=-1)
        return g0, hinv

    @staticmethod
    def _increase(rf):
        g = rf[:, 1:] - rf[:, :-1]
        min_g = g.min(dim=-1, keepdim=True).values
        r = F.relu(-min_g)
        new_g = g + r
        new_g = new_g / new_g.sum(dim=-1, keepdim=True)
        new_rf = torch.cumsum(new_g, dim=-1)
        new_rf = F.pad(new_rf, (1, 0), value=0.0)
        return new_rf

    def forward(self, feature):
        w = self.fc(feature)
        b = w.shape[0]
        g0 = self.g0.unsqueeze(0).unsqueeze(-1)
        hinv = self.hinv.unsqueeze(0).expand(b, -1, -1)
        w_exp = w.unsqueeze(-1)
        invcrf = g0 + torch.bmm(hinv, w_exp)
        invcrf = invcrf.squeeze(-1)
        invcrf = self._increase(invcrf)
        return invcrf


class LinearizationNet(nn.Module):
    """Full linearization pipeline: feature extraction + inverse CRF prediction."""

    def __init__(self, invemor_path):
        super().__init__()
        self.crf_feature_net = CrfFeatureNet()
        self.ae_invcrf_decode_net = AEInvcrfDecodeNet(invemor_path)

    @staticmethod
    def _compute_features(img):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

        edges = []
        for c in range(3):
            ch = img[:, c:c+1]
            ex = F.conv2d(ch, sobel_x, padding=1)
            ey = F.conv2d(ch, sobel_y, padding=1)
            edges.extend([ex, ey])
        edge_feat = torch.cat(edges, dim=1)

        def histogram_layer(img, max_bin):
            bins = []
            for i in range(max_bin + 1):
                h = F.relu(1.0 - torch.abs(img - i / float(max_bin)) * float(max_bin))
                bins.append(h)
            return torch.cat(bins, dim=1)

        h4 = histogram_layer(img, 4)
        h8 = histogram_layer(img, 8)
        h16 = histogram_layer(img, 16)

        return torch.cat([img, edge_feat, h4, h8, h16], dim=1)

    def forward(self, img):
        features_in = self._compute_features(img)
        feature = self.crf_feature_net(features_in)
        invcrf = self.ae_invcrf_decode_net(feature)
        return invcrf


def apply_rf_torch(x, rf):
    """Apply response function via interpolation. x: [b,c,h,w], rf: [b,1024]."""
    b, c, h, w = x.shape
    k = rf.shape[1]

    x_flat = x.reshape(b, -1)
    indices = x_flat.float() * (k - 1)

    idx0 = indices.floor().long().clamp(0, k - 1)
    idx1 = (idx0 + 1).clamp(0, k - 1)
    w1 = indices - idx0.float()
    w0 = 1.0 - w1

    v0 = torch.gather(rf, 1, idx0)
    v1 = torch.gather(rf, 1, idx1)

    result = w0 * v0 + w1 * v1
    return result.reshape(b, c, h, w)
