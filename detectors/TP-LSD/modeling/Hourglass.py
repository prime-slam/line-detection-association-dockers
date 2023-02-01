import torch
import torch.nn as nn
import torch.nn.functional as F
from .DCNv2.dcn_v2 import DCN

task = {'center':1, 'dis':4, 'line':1}
class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)

class _sigmoid():
    def __init__(self):
        pass
    def __call__(self, x):
      y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4) #将输入input张量每个元素的夹紧到区间 [min,max][min,max],避免0和1?
      return y

class outDCNconv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True):
        super(outDCNconv, self).__init__()
        self.activation = activation
        self.conv1 = DCN(in_ch, in_ch, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.convh = nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), dilation=(2, 2), stride=1, padding=(2,2))
        self.convw = nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), dilation=(2, 2), stride=1, padding=(2,2))
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.bn3 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        if self.activation:
            self.sigmoid = _sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.convh(x)))
        x = self.relu(self.bn3(self.convw(x)))
        x = self.conv3(x)
        if self.activation:
            x = self.sigmoid(x)
        return x

class double_conv(nn.Module):
    # (conv => BN => ReLU) * 2
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv_dis(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True):
        super(outconv_dis, self).__init__()
        self.activation = activation
        self.conv1 = double_conv(in_ch, in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)
        if self.activation:
            self.sigmoid = _sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.activation:
            x = self.sigmoid(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True):
        super(outconv, self).__init__()
        self.activation = activation
        self.conv1 =nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        if self.activation:
            self.sigmoid = _sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        if self.activation:
            x = self.sigmoid(x)
        return x

class MultitaskHead(nn.Module):
    def __init__(self, task_dim, input_channels):
        super(MultitaskHead, self).__init__()

        self.line_conv = nn.Conv2d(1, 1, 1)
        self.center_conv = nn.Conv2d(2, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.head_center = outDCNconv(65, task_dim['center'])

        self.head_d_dcn = DCN(65, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.head_dis = outconv_dis(64, task_dim['dis'], activation=False)
        self.head_line = outconv(64, task_dim['line'])  # , activation=False)

    def forward(self, x):
        share_feature = self.relu(self.conv1(x))
        line = self.head_line(share_feature)

        line_cat = self.tanh(self.line_conv(line))
        x = torch.cat([share_feature, line_cat], dim=1)

        center = self.head_center(x)
        # print('center', x.shape, center.shape, line.shape)
        center_cat = torch.cat([line, center], dim=1)
        center_cat = self.tanh(self.center_conv(center_cat))

        x = torch.cat([share_feature, center_cat], dim=1)
        tmp_dis = self.head_d_dcn(x)
        dis = self.head_dis(tmp_dis)

        return line, center, dis


class HourglassNet(nn.Module):
    """Hourglass model from Newell et al ECCV 2016"""
    def __init__(self, task_dim=task, inplanes=64, num_feats=128, block=Bottleneck2D, depth=4, num_stacks=2, num_blocks=1):
        super(HourglassNet, self).__init__()
        self.task_dim = task_dim
        self.inplanes = inplanes
        self.num_feats = num_feats
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats * block.expansion
        # vpts = []
        hg, res, score, score_ = [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, depth))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            # fc.append(self._make_fc(ch, ch))
            score.append(MultitaskHead(self.task_dim, ch))

            # if i < num_stacks - 1:
            #     fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
            #     score_.append(nn.Conv2d(num_classes, ch, kernel_size=1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.score = nn.ModuleList(score)
        # self.vpts = nn.ModuleList(vpts)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            line, center, dis = self.score[i](y)
            out.append({'line': line, 'center': center, 'dis': dis})
        return out

