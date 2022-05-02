from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
from models.submoduleEDNet import *
import math
import gc
import time


class MSMNet_cost(nn.Module):
    def __init__(self, second_range=12, maxdisp=192):
        super(MSMNet_cost, self).__init__()
        self.max_disp = maxdisp
        self.second_range = second_range

        self.conv1 = conv(3, 64, 7, 2)  # 1/2
        self.conv2 = ResBlock(64, 128, stride=2)  # 1/4
        self.conv3 = ResBlock(128, 256, stride=2)  # 1/8
        self.conv_compress = ResBlock(256, 32, stride=1)  # 1/8
        self.conv_redir = ResBlock(256, 32, stride=1)
        self.conv3d = nn.Sequential(
            convbn_3d(32 * 2, 32, 3, 1, 1),
            nn.ReLU(True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.conv3_1 = ResBlock(self.second_range * 2 + 32, 256)
        self.conv4 = ResBlock(256, 512, stride=2)  # 1/16
        self.conv4_1 = ResBlock(512, 512)
        self.conv5 = ResBlock(512, 512, stride=2)  # 1/32
        self.conv5_1 = ResBlock(512, 512)
        self.conv6 = ResBlock(512, 1024, stride=2)  # 1/64
        self.conv6_1 = ResBlock(1024, 1024)

        self.get_cv = GetCostVolume()
        self.iconv5 = nn.ConvTranspose2d(1024, 512, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(768, 256, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(384, 128, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(192, 64, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(96, 32, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(19, 32, 3, 1, 1)

        self.upconv5 = deconv(1024, 512)
        self.upconv4 = deconv(512, 256)
        self.upconv3 = deconv(256, 128)
        self.upconv2 = deconv(128, 64)
        self.upconv1 = deconv(64, 32)
        self.upconv0 = deconv(32, 16)

        # disparity estimation
        self.disp3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu0 = nn.ReLU(inplace=True)

        # residual learning
        self.res_submodule_2 = res_submodule_attention(scale=2, input_layer=64, value_planes=64, out_planes=32)
        self.res_submodule_1 = res_submodule_attention(scale=1, input_layer=32, value_planes=32, out_planes=32)
        self.res_submodule_0 = res_submodule_attention(scale=0, input_layer=32, value_planes=32, out_planes=32)

        self.agg = origin_agg()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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

    def forward(self, img_left, img_right):

        # split left image and right image

        conv1_l = self.conv1(img_left)  # 64 1/2
        conv2_l = self.conv2(conv1_l)  # 128 1/4
        conv3_l = self.conv3(conv2_l)  # 256 1/8

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3_r = self.conv3(conv2_r)  # 1/8

        # gwc_volume num_groups=1 和correlation效果一样
        out_corr = build_gwc_volume(conv3_l, conv3_r, maxdisp=48, num_groups=1, step=2)
        pr3_o = self.agg(out_corr)

        # step2

        disp_range = get_disp_range_samples(cur_disp=torch.squeeze(pr3_o / 8, dim=1), ndisp=self.second_range,
                                            disp_inteval_pixel=1,
                                            dtype=pr3_o.dtype,
                                            device=pr3_o.device,
                                            shape=[pr3_o.shape[0], pr3_o.shape[2], pr3_o.shape[3]],
                                            max_disp=self.max_disp,
                                            using_ns=None,
                                            ns_size=None)

        cost = self.get_cv(conv3_l, conv3_r,
                           disp_range_samples=disp_range,
                           ndisp=self.second_range)

        conv_compress_left = self.conv_compress(conv3_l)
        conv_compress_right = self.conv_compress(conv3_r)

        cost_volume = self.get_cv(conv_compress_left, conv_compress_right, disp_range_samples=disp_range,
                                  ndisp=self.second_range, mode="concat")
        cost_volume = self.conv3d(cost_volume)
        cost_volume = torch.squeeze(cost_volume, dim=1)
        out_conv3a_redir = self.conv_redir(conv3_l)
        in_conv3b = torch.cat((out_conv3a_redir, cost, cost_volume), dim=1)  # self.second_range+self.second_range+32

        conv3b = self.conv3_1(in_conv3b)  # 256
        conv4a = self.conv4(conv3b)
        conv4b = self.conv4_1(conv4a)  # 512 1/16
        conv5a = self.conv5(conv4b)
        conv5b = self.conv5_1(conv5a)  # 512 1/32
        conv6a = self.conv6(conv5b)
        conv6b = self.conv6_1(conv6a)  # 1024 1/64

        upconv5 = self.upconv5(conv6b)  # 512 1/32
        concat5 = torch.cat((upconv5, conv5b), dim=1)  # 1024 1/32
        iconv5 = self.iconv5(concat5)  # 512

        upconv4 = self.upconv4(iconv5)  # 256 1/16
        concat4 = torch.cat((upconv4, conv4b), dim=1)  # 768 1/16
        iconv4 = self.iconv4(concat4)  # 256 1/16

        upconv3 = self.upconv3(iconv4)  # 128 1/8
        concat3 = torch.cat((upconv3, conv3b), dim=1)  # 128+256=384 1/8
        iconv3 = self.iconv3(concat3)  # 128
        pr3 = self.disp3(iconv3)
        pr3 = self.relu3(pr3 + pr3_o)

        upconv2 = self.upconv2(iconv3)  # 64 1/4
        concat2 = torch.cat((upconv2, conv2_l), dim=1)  # 192 1/4
        iconv2 = self.iconv2(concat2)
        # pr2 = self.upflow3to2(pr3)
        pr2 = F.interpolate(pr3, size=(pr3.size()[2] * 2, pr3.size()[3] * 2), mode='bilinear')
        res2 = self.res_submodule_2(img_left, img_right, pr2, iconv2)
        pr2 = pr2 + res2
        pr2 = self.relu2(pr2)

        upconv1 = self.upconv1(iconv2)  # 32 1/2
        concat1 = torch.cat((upconv1, conv1_l), dim=1)  # 32+64=96
        iconv1 = self.iconv1(concat1)  # 32 1/2
        # pr1 = self.upflow2to1(pr2)
        pr1 = F.interpolate(pr2, size=(pr2.size()[2] * 2, pr2.size()[3] * 2), mode='bilinear')
        res1 = self.res_submodule_1(img_left, img_right, pr1, iconv1)  #
        pr1 = pr1 + res1
        pr1 = self.relu1(pr1)

        upconv1 = self.upconv0(iconv1)  # 16 1
        concat0 = torch.cat((upconv1, img_left), dim=1)  # 16+3=19 1
        iconv0 = self.iconv0(concat0)  # 16 1
        # pr0 = self.upflow1to0(pr1)
        pr0 = F.interpolate(pr1, size=(pr1.size()[2] * 2, pr1.size()[3] * 2), mode='bilinear')
        res0 = self.res_submodule_0(img_left, img_right, pr0, iconv0)
        pr0 = pr0 + res0
        pr0 = self.relu0(pr0)
        if self.training:
            return [pr0, pr1, pr2, pr3, pr3_o]
        else:
            return pr0


class origin_agg(nn.Module):
    def __init__(self):
        super(origin_agg, self).__init__()
        self.conv3_1_o = ResBlock(24, 128)
        self.conv4_o = ResBlock(128, 256, stride=2)  # 1/16
        self.conv4_1_o = ResBlock(256, 256)
        self.conv5_o = ResBlock(256, 256, stride=2)  # 1/32
        self.conv5_1_o = ResBlock(256, 256)
        self.conv6_o = ResBlock(256, 512, stride=2)  # 1/64
        self.conv6_1_o = ResBlock(512, 512)
        self.iconv5_o = nn.ConvTranspose2d(512, 256, 3, 1, 1)
        self.iconv4_o = nn.ConvTranspose2d(384, 128, 3, 1, 1)
        self.iconv3_o = nn.ConvTranspose2d(192, 64, 3, 1, 1)
        self.upconv5_o = deconv(512, 256)
        self.upconv4_o = deconv(256, 128)
        self.upconv3_o = deconv(128, 64)
        self.disp3_o = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, out_corr):
        conv3b = self.conv3_1_o(out_corr)  # 128
        conv4a = self.conv4_o(conv3b)
        conv4b = self.conv4_1_o(conv4a)  # 256 1/16
        conv5a = self.conv5_o(conv4b)
        conv5b = self.conv5_1_o(conv5a)  # 256 1/32
        conv6a = self.conv6_o(conv5b)
        conv6b = self.conv6_1_o(conv6a)  # 512 1/64

        upconv5 = self.upconv5_o(conv6b)  # 256 1/32
        concat5 = torch.cat((upconv5, conv5b), dim=1)  # 512 1/32
        iconv5 = self.iconv5_o(concat5)  # 256

        upconv4 = self.upconv4_o(iconv5)  # 128 1/16
        concat4 = torch.cat((upconv4, conv4b), dim=1)  # 384 1/16
        iconv4 = self.iconv4_o(concat4)  # 128 1/16

        upconv3 = self.upconv3_o(iconv4)  # 64 1/8
        concat3 = torch.cat((upconv3, conv3b), dim=1)  # 192 1/8
        iconv3 = self.iconv3_o(concat3)  # 64
        pr3_o = self.disp3_o(iconv3)
        pr3_o = self.relu3(pr3_o)
        return pr3_o


if __name__ == '__main__':
    import time

    model = MSMNet_cost(second_range=12).cuda()
    model.eval()
    input = torch.randn(1, 3, 384, 1280).cuda()
    # input2 = torch.randn(1, 3, 384, 1280).cuda()
    from thop.profile import profile

    flops, params = profile(model, inputs=(input, input))
    print('Number of Params: %.5f' % (params / 1e6))
    print('Number of GFLOPs: %.5f' % (flops / 1e9))

    # print('Number of Params: %.5f' % (params / 1e6))
    # print('Number of GFLOPs: %.5f' % (flops / 1e9))
    total_time = 0

    for i in range(10):
        start_time = time.time()
        out = model(input, input)
        end_time = time.time()
        total_time += end_time - start_time
    print('avg time is {:.3f}'.format(total_time / 10))
