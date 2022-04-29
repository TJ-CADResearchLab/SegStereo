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
        concat_feature = self.lastconv(gwc_feature)
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


class AttNet(nn.Module):
    def __init__(self, maxdisp):
        super(AttNet, self).__init__()
        self.maxdisp = maxdisp
        self.num_groups = 40
        self.concat_channels = 32
        self.feature_extraction = feature_extraction(concat_feature_channel=self.concat_channels)
        self.patch_l1 = nn.Conv3d(8, 8, kernel_size=(1, 3, 3), stride=1, dilation=1, groups=8, padding=(0, 1, 1),
                                  bias=False)
        self.patch_l2 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, dilation=2, groups=16, padding=(0, 2, 2),
                                  bias=False)
        self.patch_l3 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, dilation=3, groups=16, padding=(0, 3, 3),
                                  bias=False)

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
        self.gamma_s3 = nn.Parameter(torch.zeros(1))
        self.beta_s3 = nn.Parameter(torch.zeros(1))
        self.spatial_transformer = SpatialTransformer()
        self.uniform_sampler = UniformSampler()
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

    def generate_disparity_samples(self, min_disparity, max_disparity, sample_count=10):
        """
        Description:    Generates "sample_count" number of disparity samples from the
                                                            search range (min_disparity, max_disparity)
                        Samples are generated by uniform sampler

        Args:
            :min_disparity: LowerBound of the disaprity search range.
            :max_disparity: UpperBound of the disparity search range.
            :sample_count: Number of samples to be generated from the input search range.

        Returns:
            :disparity_samples:
        """
        disparity_samples = self.uniform_sampler(min_disparity, max_disparity, sample_count)

        disparity_samples = torch.cat((torch.floor(min_disparity), disparity_samples, torch.ceil(max_disparity)),
                                      dim=1).long()  # disparity level = sample_count + 2
        return disparity_samples

    def cost_volume_generator(self, left_input, right_input, disparity_samples, model='concat', num_groups=40):
        """
        Description: Generates cost-volume using left image features, disaprity samples
                                                            and warped right image features.
        Args:
            :left_input: Left Image fetaures
            :right_input: Right Image features
            :disparity_samples: Disaprity samples
            :model : concat or group correlation

        Returns:
            :cost_volume:
            :disaprity_samples:
        """

        right_feature_map, left_feature_map = self.spatial_transformer(left_input,
                                                                       right_input, disparity_samples)
        disparity_samples = disparity_samples.unsqueeze(1).float()
        if model == 'concat':
            cost_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        else:
            cost_volume = groupwise_correlation_4D(left_feature_map, right_feature_map, num_groups)

        return cost_volume, disparity_samples

    def generate_search_range(self, sample_count, input_min_disparity, input_max_disparity, scale):
        """
        Description:    Generates the disparity search range.

        Returns:
            :min_disparity: Lower bound of disparity search range
            :max_disparity: Upper bound of disaprity search range.
        """

        min_disparity = torch.clamp(input_min_disparity - torch.clamp((
                sample_count - input_max_disparity + input_min_disparity), min=0) / 2.0, min=0,
                                    max=self.maxdisp // (2 ** scale) - 1)
        max_disparity = torch.clamp(input_max_disparity + torch.clamp(
            sample_count - input_max_disparity + input_min_disparity, min=0) / 2.0, min=0,
                                    max=self.maxdisp // (2 ** scale) - 1)

        return min_disparity, max_disparity

    def forward(self, left, right, valid=False):
        features_left = self.feature_extraction(
            left)  # gwc_feature [b, 320, 1/4h , 1/4w]; concat_feature [b, 32, 1/4h, 1/4w]
        features_right = self.feature_extraction(right)

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups, step=2)  # [batch, c, disp, h, w] = [b, 40, 24 ,1/4h, 1/4w]

        patch_l1 = self.patch_l1(gwc_volume[:, :8])
        patch_l2 = self.patch_l2(gwc_volume[:, 8:24])
        patch_l3 = self.patch_l3(gwc_volume[:, 24:40])
        patch_volume = torch.cat((patch_l1, patch_l2, patch_l3), dim=1)  # [batch, 40, 24, 1/4h, 1/4w]

        cost_attention = self.dres1_att(patch_volume)  # [batch, 16, 24, 1/4h, 1/4w]
        cost_attention = self.dres2_att(cost_attention)
        att_weights = self.classif_att(cost_attention)  # att [batch, 1, 24, 1/4h, 1/4w]
        pred_attention_pos = F.softmax(att_weights.squeeze(1), dim=1)   # att_pos [batch, 24, 1/4h, 1/4w]
        pred_attention = disparity_regression(pred_attention_pos, self.maxdisp // 4,
                                              step=2, keepdim=True)  # predatt [batch, 1, 1/4h , 1/4w]

        pred2_s4_cur = pred_attention.detach()
        pred2_v_s4 = disparity_variance(pred_attention_pos, self.maxdisp // 4, pred2_s4_cur, step=2)  # get the variance
        pred2_v_s4 = pred2_v_s4.sqrt()

        mindisparity_s3 = pred2_s4_cur - (self.gamma_s3 + 1) * pred2_v_s4 - self.beta_s3
        maxdisparity_s3 = pred2_s4_cur + (self.gamma_s3 + 1) * pred2_v_s4 + self.beta_s3

        mindisparity_s3_1, maxdisparity_s3_1 = self.generate_search_range(10 + 1, mindisparity_s3, maxdisparity_s3,
                                                                          scale=2)
        disparity_samples = self.generate_disparity_samples(mindisparity_s3_1, maxdisparity_s3_1,
                                                            10).float()  # dis_sam [batch, 10+2, 1/4h, 1/4w]

        ac_volume, _ = self.cost_volume_generator(features_left["concat_feature"],
                                                  features_right["concat_feature"], disparity_samples,
                                                  'concat')  # ac [batch, 64, 12, 1/4h, 1/4w]

        cost0 = self.dres0(ac_volume)
        cost0 = self.dres1(cost0) + cost0  # cost0 [batch, 32, 12, 1/4h, 1/4w]

        cost0_c = self.classif0(cost0)  # cost0_c [batch, 1, 12, 1/4h, 1/4w]
        pred0_pos = F.softmax(cost0_c.squeeze(1), dim=1)  # pre_pos [batch, 12, 1/4h, 1/4w]
        pred0 = torch.sum(pred0_pos * disparity_samples, dim=1, keepdim=True)
        pred0_cur = pred0.detach()
        pred0_u = disparity_variance_confidence(pred0_pos, disparity_samples, pred0_cur)
        pix_att0 = F.sigmoid(pred0_u)  # pix_att0 [batch, 1, 1/4h, 1/4w]


        out1 = self.dres2(cost0 * pix_att0.unsqueeze(1) * pred0_pos.unsqueeze(1))
        cost1 = self.classif1(out1)  # cost1 [batch, 1, 12, 1/4h, 1/4w]
        pred1_pos = F.softmax(cost1.squeeze(1), dim=1)  # pre1_pos [batch, 12, 1/4h, 1/4w]
        pred1 = torch.sum(pred1_pos * disparity_samples, dim=1, keepdim=True)
        pred1_cur = pred1.detach()
        pred1_u = disparity_variance_confidence(pred1_pos, disparity_samples, pred1_cur)
        pix_att1 = F.sigmoid(pred1_u)  # pix_att1 [batch, 1, 1/4h, 1/4w]


        out2 = self.dres3(out1 * pix_att1.unsqueeze(1) * pred1_pos.unsqueeze(1))
        cost2 = self.classif2(out2)  # cost2 [batch, 1, 12, 1/4h, 1/4w]

        maxdisparity_f = F.upsample(maxdisparity_s3 * 4, [left.size()[2], left.size()[3]],
                                    mode='bilinear', align_corners=True)
        mindisparity_f = F.upsample(mindisparity_s3 * 4, [left.size()[2], left.size()[3]],
                                    mode='bilinear',
                                    align_corners=True)

        mindisparity_f, maxdisparity_f = self.generate_search_range(46 + 1, mindisparity_f,
                                                                    maxdisparity_f, scale=0)
        disparity_samples_f = self.generate_disparity_samples(mindisparity_f, maxdisparity_f,
                                                              46).float()

        if self.training:
            att_weights = F.upsample(att_weights, [24*4, left.size()[2], left.size()[3]], mode='trilinear') # att [batch, 1, 96, h, w]
            att_weights=torch.squeeze(att_weights,1)
            att_pred =F.softmax(att_weights,dim=1)
            att_pred = disparity_regression(att_pred, self.maxdisp,step=2)


            cost0_c=F.upsample(cost0_c, [12*4, left.size()[2], left.size()[3]], mode='trilinear')   # cost0_c [batch, 1, 48, h, w]
            cost0_c=torch.squeeze(cost0_c,1)    # cost0_c [batch, 48, h, w]
            pred0_f=F.softmax(cost0_c,dim=1)
            pred0_f=torch.sum(pred0_f*disparity_samples_f,dim=1,keepdim=False)

            cost1=F.upsample(cost1, [12*4, left.size()[2], left.size()[3]], mode='trilinear')   # cost1 [batch, 1, 48, h, w]
            cost1=torch.squeeze(cost1,1)    # cost0_c [batch, 48, h, w]
            pred1_f=F.softmax(cost1,dim=1)
            pred1_f=torch.sum(pred1_f*disparity_samples_f,dim=1,keepdim=False)

            cost2=F.upsample(cost2, [12*4, left.size()[2], left.size()[3]], mode='trilinear')   # cost2 [batch, 1, 48, h, w]
            cost2=torch.squeeze(cost2,1)    # cost2 [batch, 48, h, w]
            pred2_f=F.softmax(cost2,dim=1)
            pred2_f=torch.sum(pred2_f*disparity_samples_f,dim=1,keepdim=False)

            return [att_pred, pred0_f, pred1_f, pred2_f]
        # elif valid:
        #     return [pred2, pix_att0.squeeze(1), pix_att1.squeeze(1), disparity_samples]
        else:

            cost2=F.upsample(cost2, [12*4, left.size()[2], left.size()[3]], mode='trilinear')   # cost2 [batch, 1, 48, h, w]
            cost2=torch.squeeze(cost2,1)    # cost2 [batch, 48, h, w]
            pred2_f=F.softmax(cost2,dim=1)
            pred2_f=torch.sum(pred2_f*disparity_samples_f,dim=1,keepdim=False)
            return [pred2_f]


def att(d):
    return AttNet(d)


if __name__ == "__main__":
    model = AttNet(maxdisp=192)
    left = torch.rand([1, 3, 256, 512])
    right = torch.rand([1, 3, 256, 512])
    model.train()
    out = model(left, right)
