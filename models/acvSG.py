from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
import gc
import time


class feature_extraction(nn.Module):
    def __init__(self, concat_feature_channel=32):
        super(feature_extraction, self).__init__()

        self.inplanes = 16
        self.firstconv = nn.Sequential(convbn(3, 16, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(16, 16, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(16, 16, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 2, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
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

    def forward(self, x, return_feature=False):
        l0 = self.firstconv(x)  # b,16,h,w
        l1 = self.layer1(l0)  # b,32,1/2h,1/2w
        l2 = self.layer2(l1)  # b,64,1/4h,1/4w
        l3 = self.layer3(l2)  # b,128,1/4h,1/4w
        l4 = self.layer4(l3)  # b,128,1/4h,1/4w

        gwc_feature = torch.cat((l2, l3, l4), dim=1)
        concat_feature = self.lastconv(gwc_feature)
        if return_feature:
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature, "teacher_feature": [l2, l1, l0]}
        else:
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}


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
        self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

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
        conv4 = self.attention_block(conv4)

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
                                  nn.ReLU(inplace=True),
                                  convbn(seg_channel + 2, simple_nums * 3, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True),
                                  convbn(simple_nums * 3, simple_nums * 3, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True))

    def forward(self, seg_feature, error, disp):
        disp = torch.unsqueeze(disp, dim=1)
        # seg_feature [b,c,h,w] error [b,1,h,w] disp [b,1,h,w]
        sample = self.conv(torch.cat([seg_feature, error, disp], dim=1))  # sample [b,node*3,h,w]
        sample, weight = sample.split([2 * self.simple_nums, self.simple_nums],
                                      dim=1)  # sample [b,node*2,h,w]  weight [b,node,h,w]
        sample = sample.view(sample.size()[0], sample.size()[1] // 2, 2, sample.size()[2], sample.size()[3])
        sample = sample.permute(0, 1, 3, 4, 2)  # sample [b,node,h,w,2]
        disp_ref = torch.zeros([sample.size()[0], sample.size()[1], sample.size()[2], sample.size()[3]],
                               device=sample.device)
        for i in range(sample.size()[1]):
            disp_ref[:, i, :, :] = resamplexy(disp, sample[:, i, :, :, :]).squeeze(1)
        if sample.size()[1] == 1:
            disp_ref = torch.squeeze(disp_ref, dim=1)
        else:
            disp_ref = torch.sum(disp_ref * F.softmax(weight, dim=1), dim=1, keepdim=False)
        return disp_ref


class ACVSGNet(nn.Module):
    def __init__(self, maxdisp, only_train_seg=False):
        super(ACVSGNet, self).__init__()
        self.maxdisp = maxdisp
        self.num_groups = 40
        self.concat_channels = 32
        self.feature_extraction = feature_extraction(concat_feature_channel=self.concat_channels)
        self.only_train_seg = only_train_seg
        if self.only_train_seg:
            for p in self.feature_extraction.parameters():
                p.requires_grad = False
        self.patch_l1 = nn.Conv3d(8, 8, kernel_size=(1, 3, 3), stride=1, dilation=1, groups=8, padding=(0, 1, 1),
                                  bias=False)
        self.patch_l2 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, dilation=2, groups=16, padding=(0, 2, 2),
                                  bias=False)
        self.patch_l3 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, dilation=3, groups=16, padding=(0, 3, 3),
                                  bias=False)

        self.seghead0 = seghead(64)
        self.seghead1 = seghead(32)
        self.seghead2 = seghead(16)
        self.refine0 = refine(64, simple_nums=9)
        self.refine1 = refine(32, simple_nums=9)
        self.refine2 = refine(16, simple_nums=9)

        self.dres1_att = nn.Sequential(convbn_3d(40, 16, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn_3d(16, 16, 3, 1, 1))
        self.dres2_att = hourglass(16)
        self.classif_att = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.dres0 = nn.Sequential(convbn_3d(self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
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

    def forward(self, left, right, refine_mode=False):

        if self.only_train_seg:  # only train seg head
            with torch.no_grad():
                features_left = self.feature_extraction(left, return_feature=True)
                features_right = self.feature_extraction(right, return_feature=True)
            teacher_feature_left = features_left["teacher_feature"].detach()
            segfea_left0 = self.seghead0(teacher_feature_left[0])  # [b,64,1/4h,1/4w]
            segfea_left1 = self.seghead1(teacher_feature_left[1])  # [b,32,1/2h,1/2w]
            segfea_left2 = self.seghead2(teacher_feature_left[2])  # [b,16,h,w]

            teacher_feature_right = features_right["teacher_feature"].detach()
            segfea_right0 = self.seghead0(teacher_feature_right[0])  # [b,64,1/4h,1/4w]
            segfea_right1 = self.seghead1(teacher_feature_right[1])  # [b,32,1/2h,1/2w]
            segfea_right2 = self.seghead2(teacher_feature_right[2])  # [b,16,h,w]
            return teacher_feature_left, teacher_feature_right, [segfea_left0, segfea_left1, segfea_left2], [
                segfea_right0, segfea_right1, segfea_right2]

        features_left = self.feature_extraction(left, return_feature=refine_mode)
        features_right = self.feature_extraction(right)
        # gwc_feature [b, 320, 1/4h , 1/4w]; concat_feature [b, 32, 1/4h, 1/4w];
        # teacher_feature [[b, 64, 1/4h, 1/4w],   [b,32,1/2h,1/2w],   [b,16,h,w]]

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        patch_l1 = self.patch_l1(gwc_volume[:, :8])
        patch_l2 = self.patch_l2(gwc_volume[:, 8:24])
        patch_l3 = self.patch_l3(gwc_volume[:, 24:40])
        patch_volume = torch.cat((patch_l1, patch_l2, patch_l3), dim=1)
        concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                            self.maxdisp // 4)
        cost_attention = self.dres1_att(patch_volume)

        cost_attention = self.dres2_att(cost_attention)  # [b, 16, 48, 1/4h, 1/4w]

        att_weights = self.classif_att(cost_attention)  # [b, 1, 48, 1/4h, 1/4w]
        ac_volume = att_weights * concat_volume  # [b, 64, 48, 1/4h, 1/4w]

        cost0 = self.dres0(ac_volume)  # [b, 32, 48, 1/4h, 1/4w]
        cost0 = self.dres1(cost0) + cost0  # [b, 32, 48, 1/4h, 1/4w]
        out1 = self.dres2(cost0)  # [b, 32, 48, 1/4h, 1/4w]
        out2 = self.dres3(out1)  # [b, 32, 48, 1/4h, 1/4w]

        if self.training:

            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)

            cost_attention = F.upsample(att_weights, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost_attention = torch.squeeze(cost_attention, 1)
            pred_attention = F.softmax(cost_attention, dim=1)
            pred_attention = disparity_regression(pred_attention, self.maxdisp)

            cost0 = F.upsample(cost0, [self.maxdisp // 4, left.size()[2] // 4, left.size()[3] // 4], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0_pos = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0_pos, self.maxdisp // 4)

            cost1 = F.upsample(cost1, [self.maxdisp // 2, left.size()[2] // 2, left.size()[3] // 2], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1_pos = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1_pos, self.maxdisp // 2)

            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2_pos = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2_pos, self.maxdisp)

            if refine_mode:
                teacher_feature_left = features_left["teacher_feature"].detach()
                segfea_left0 = self.seghead0(teacher_feature_left[0])  # [b,64,1/4h,1/4w]
                segfea_left1 = self.seghead1(teacher_feature_left[1])  # [b,32,1/2h,1/2w]
                segfea_left2 = self.seghead2(teacher_feature_left[2])  # [b,16,h,w]

                error0 = disparity_variance(pred0_pos, self.maxdisp // 4, pred0.unsqueeze(1))  # get the variance
                error0 = error0.sqrt().detach()
                pred0_ref = self.refine0(segfea_left0, error0, pred0)

                error1 = disparity_variance(pred1_pos, self.maxdisp // 2, pred1.unsqueeze(1))  # get the variance
                error1 = error1.sqrt().detach()
                pred1_ref = self.refine1(segfea_left1, error1, pred1)

                error2 = disparity_variance(pred2_pos, self.maxdisp, pred2.unsqueeze(1))  # get the variance
                error2 = error2.sqrt().detach()
                pred2_ref = self.refine2(segfea_left2, error2, pred2)

                teacher_feature_right = features_right["teacher_feature"].detach()
                segfea_right0 = self.seghead0(teacher_feature_right[0])  # [b,64,1/4h,1/4w]
                segfea_right1 = self.seghead1(teacher_feature_right[1])  # [b,32,1/2h,1/2w]
                segfea_right2 = self.seghead2(teacher_feature_right[2])  # [b,16,h,w]

                return [pred_attention, pred0, pred1, pred2, pred0_ref, pred1_ref,
                        pred2_ref], [teacher_feature_left, teacher_feature_right, [segfea_left0, segfea_left1,
                                                                                   segfea_left2], [segfea_right0,
                                                                                                   segfea_right1,
                                                                                                   segfea_right2]]
            else:
                return [pred_attention, pred0, pred1, pred2]

        else:
            cost2 = self.classif2(out2)
            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2_pos = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2_pos, self.maxdisp)

            if refine_mode:
                teacher_feature_left = features_left["teacher_feature"].detach()
                segfea2 = self.seghead2(teacher_feature_left[2])  # [b,16,h,w]
                error2 = disparity_variance(pred2_pos, self.maxdisp, pred2.unsqueeze(1))  # get the variance
                error2 = error2.sqrt().detach()
                pred2_ref = self.refine2(segfea2, error2, pred2)

                return [pred2, pred2_ref]
            else:
                return [pred2]


def acvsg(d):
    return ACVSGNet(d)


if __name__ == "__main__":
    model = ACVSGNet(maxdisp=192, only_train_seg=True)
    left = torch.rand([1, 3, 256, 512])
    right = torch.rand([1, 3, 256, 512])
    model.train()
    out = model(left, right, refine_mode=True)
    for p in model.feature_extraction.parameters():
        print(p.requires_grad)
# print(out[0].shape,out[1].shape,out[2].shape,out[3].shape)
