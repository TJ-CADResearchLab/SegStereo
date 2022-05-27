import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:  # self.features是一个列表，列表中的每一项是多个卷积层，用来将输入的特征图自适应平均池化到指定大小，然后用1*1的卷积降低维度
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:  # f(x)表示对输入的特征图执行sele.features中每个元素所指定的卷积操作，从而将输入特征图分别变为512*1*1,512*2*2，512*3*3,512*6*6的特征图
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear',
                                     align_corners=True))  # 将特征图的尺寸利用上采样变为512*60*60,512*60*60,512*60*60,512*60*60
        return torch.cat(out, 1)  # 此时out有5个元素，分别为2048*60*60,512*60*60，512*60*60，512*60*60，512*60*60


class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        models.BatchNorm = BatchNorm
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins,
                           BatchNorm)  # fea_dim为2048，bins为[1,2,3,6]表示生成的特征图的尺寸，
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        x_size = x.size()  # torch.Size([4, 3, 473, 473])
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0  # 对输入图片的尺寸有限制，（473-1）%8=0
        h = int(
            (x_size[2] - 1) / 8 * self.zoom_factor + 1)  # （(x_size[2] - 1) / 8 * self.zoom_factor + 1）=（472/8）×1+1=60
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)  # 60
        x = self.layer0(x)  # size为(4,128,119,119) 下采样了4倍，因为resnet的输入就是下采样，然后layer1会用Maxpool下采样
        x = self.layer1(x)  # layer1中只有常规卷积，没有空洞卷积和下采样，输出为(4,256,119,119)
        x = self.layer2(x)  # layer2用卷积层stride=（2,2）下采样一次，输出为（4,612,60,60）
        x_tmp = self.layer3(x)  # 输出为（4,1024,60,60），没有下采样，使用了6个（2*2）的空洞卷积来扩大感受野
        x = self.layer4(x_tmp)  # 使用了3个（4*4）的空洞卷积来扩大感受野，输出为（4,2048,60,60）
        if self.use_ppm:
            x = self.ppm(
                x)  # 输入是（4,2048,60,60）,然后使用自适应全局平均池化分别得到size为1*1,2*2,3*3,6*6的feature map,这些featuremap都转化为512通道,然后将x与所有的feature map做concat为4096通道
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    input = torch.rand(4, 3, 473, 473).cuda()
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=8, use_ppm=True,
                   pretrained=False).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output.size())