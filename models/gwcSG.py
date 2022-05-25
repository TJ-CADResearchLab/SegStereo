from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature,"teacher_feature":l4}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature,"teacher_feature":concat_feature}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6

class seghead(nn.Module):
    def __init__(self, in_channels):
        super(seghead, self).__init__()
        self.conv = nn.Sequential(convbn(in_channels, in_channels, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True),
                                  convbn(in_channels, in_channels, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True),
                                  convbn(in_channels, in_channels, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class refine(nn.Module):
    def __init__(self, seg_channel, simple_nums=1):
        super(refine, self).__init__()
        self.simple_nums = simple_nums
        self.conv = nn.Sequential(convbn(seg_channel + 2, seg_channel + 2, 3, 1, 1, 1),
                                  nn.Sigmoid(),
                                  convbn(seg_channel + 2, simple_nums * 3, 3, 1, 1, 1),
                                  nn.Sigmoid(),
                                  convbn(simple_nums * 3, simple_nums * 3, 3, 1, 1, 1),
                                  nn.Sigmoid())

    def forward(self, seg_feature, error, disp):
        disp = torch.unsqueeze(disp, dim=1)
        # seg_feature [b,c,h,w] error [b,1,h,w] disp [b,1,h,w]
        sample = self.conv(torch.cat([seg_feature, error, disp], dim=1))  # sample [b,node*3,h,w]

        sample, weight = sample.split([2 * self.simple_nums, self.simple_nums],
                                      dim=1)  # sample [b,node*2,h,w]  weight [b,node,h,w]
        sample = sample.view(sample.size()[0], sample.size()[1] // 2, 2, sample.size()[2], sample.size()[3])
        sample = sample.permute(0, 1, 3, 4, 2)  # sample [b,node,h,w,2]
        sample=(sample-0.5)*10
        disp_ref = torch.zeros([sample.size()[0], sample.size()[1], sample.size()[2], sample.size()[3]],
                               device=sample.device)
        for i in range(sample.size()[1]):
            disp_ref[:, i, :, :] = resamplexy(disp, sample[:, i, :, :, :]).squeeze(1)
        if sample.size()[1] == 1:
            disp_ref = torch.squeeze(disp_ref, dim=1)
        else:
            disp_ref = torch.sum(disp_ref * F.softmax(weight, dim=1), dim=1, keepdim=False)
        return disp_ref

class GwcSGNet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=False):
        super(GwcSGNet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume

        self.num_groups = 40

        self.seghead = seghead(128)
        self.refine = refine(128, simple_nums=1)

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right,refine_mode=False):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3_pos = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3_pos, self.maxdisp)

            if refine_mode:
                teacher_feature_left = features_left["teacher_feature"].detach()
                segfea_left = self.seghead(teacher_feature_left)  # [b,128,1/4h,1/4w]
                segfea_left = F.upsample(segfea_left, scale_factor=4, mode='bilinear',align_corners=True)
                error3 = disparity_variance(pred3_pos, self.maxdisp, pred3.unsqueeze(1))  # get the variance
                error3 = error3.sqrt()
                pred3_ref = self.refine(segfea_left, error3.detach(), pred3.detach())

                teacher_feature_right=features_right["teacher_feature"].detach()
                segfea_right=self.seghead(teacher_feature_right)

                return [pred0, pred1, pred2, pred3,pred3_ref],[[teacher_feature_left],[teacher_feature_right],[segfea_left],[segfea_right]]
            else:
                return [pred0,pred1,pred2,pred3]
        else:
            cost3 = self.classif3(out3)
            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3_pos = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3_pos, self.maxdisp)

            if refine_mode:
                teacher_feature_left = features_left["teacher_feature"].detach()
                segfea_left = self.seghead(teacher_feature_left)  # [b,128,1/4h,1/4w]
                segfea_left = F.upsample(segfea_left, scale_factor=4,
                                         mode='bilinear', align_corners=True)
                error3 = disparity_variance(pred3_pos, self.maxdisp, pred3.unsqueeze(1))  # get the variance
                error3 = error3.sqrt()
                pred3_ref = self.refine(segfea_left, error3.detach(), pred3.detach())

                return [pred3,pred3_ref]
            else:
                return [pred3]


def GwcSGNet_G(d):
    return GwcSGNet(d, use_concat_volume=False)


def GwcSGNet_GC(d):
    return GwcSGNet(d, use_concat_volume=True)


if __name__ == "__main__":
    model = GwcSGNet(maxdisp=192)
    left = torch.rand([1, 3, 256, 512])
    right = torch.rand([1, 3, 256, 512])
    model.train()
    out = model(left, right, refine_mode=True)
    for name, p in model.named_parameters():
        print(name)