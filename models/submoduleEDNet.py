from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np


def conv(in_planes, out_planes, kernel_size=3, stride=1, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class GetCostVolume(nn.Module):
    def __init__(self):
        super(GetCostVolume, self).__init__()

    def forward(self, x, y, disp_range_samples, ndisp,mode="corrlation"):
        assert (x.is_contiguous() == True)

        bs, channels, height, width = x.size()
        if mode=="corrlation":

            cost = x.new().resize_(bs, channels , ndisp, height, width).zero_()
        else:
            cost=x.new().resize_(bs, channels*2 , ndisp, height, width).zero_()
        # cost = y.unsqueeze(2).repeat(1, 2, ndisp, 1, 1) #(B, 2C, D, H, W)

        mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                                 torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)
        mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
        mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

        cur_disp_coords_y = mh
       # print(ndisp)
       # print(mw.shape,disp_range_samples.shape)
        cur_disp_coords_x = mw - disp_range_samples

        coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
        coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
        grid = torch.stack([coords_x, coords_y], dim=4)   #(B, D, H, W, 2)

        if mode=="concat":
            cost[:, x.size()[1]:, :, :, :] = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                                                           padding_mode='zeros',align_corners=True).view(bs, channels, ndisp, height, width)
            tmp = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1) #(B, C, D, H, W)
            cost[:, :x.size()[1], :, :, :] = tmp
        elif mode=="corrlation":
            cost[:, :, :, :, :] = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                                                           padding_mode='zeros',align_corners=True).view(bs, channels, ndisp, height, width)
            tmp = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1) #(B, C, D, H, W)
            cost=(cost*tmp).mean(dim=1)
            cost=torch.squeeze(cost,dim=1)
        else:
            AssertionError

        return cost

def get_cur_disp_range_samples(cur_disp, ndisp, disp_inteval_pixel, shape, ns_size, using_ns=False, max_disp=192.0):
    #shape, (B, H, W)
    #cur_disp: (B, H, W)
    #return disp_range_samples: (B, D, H, W)
    if not using_ns:
        cur_disp_min = (cur_disp - ndisp / 2 * disp_inteval_pixel)  # (B, H, W)
        cur_disp_max = (cur_disp + ndisp / 2 * disp_inteval_pixel)
        # cur_disp_min = (cur_disp - ndisp / 2 * disp_inteval_pixel).clamp(min=0.0)   #(B, H, W)
        # cur_disp_max = (cur_disp_min + (ndisp - 1) * disp_inteval_pixel).clamp(max=max_disp)

        assert cur_disp.shape == torch.Size(shape), "cur_disp:{}, input shape:{}".format(cur_disp.shape, shape)
        new_interval = (cur_disp_max - cur_disp_min) / (ndisp - 1)  # (B, H, W)

        disp_range_samples = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp, device=cur_disp.device,
                                                                      dtype=cur_disp.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1,
                                                                                                   1) * new_interval.unsqueeze(1))
    else:
        #using neighbor region information to help determine the range.
        #consider the maximum and minimum values ​​in the region.
        assert cur_disp.shape == torch.Size(shape), "cur_disp:{}, input shape:{}".format(cur_disp.shape, shape)
        B, H, W = cur_disp.shape
        cur_disp_smooth = F.interpolate((cur_disp / 4.0).unsqueeze(1),
                                        [H // 4, W // 4], mode='bilinear', align_corners=Align_Corners_Range).squeeze(1)
        #get minimum value
        disp_min_ns = torch.abs(F.max_pool2d(-cur_disp_smooth, stride=1, kernel_size=ns_size, padding=ns_size // 2))    # (B, 1/4H, 1/4W)
        #get maximum value
        disp_max_ns = F.max_pool2d(cur_disp_smooth, stride=1, kernel_size=ns_size, padding=ns_size // 2)

        disp_pred_inter = torch.abs(disp_max_ns - disp_min_ns)    #(B, 1/4H, 1/4W)
        disp_range_comp = (ndisp//4 * disp_inteval_pixel - disp_pred_inter).clamp(min=0) / 2.0  #(B, 1/4H, 1/4W)

        cur_disp_min = (disp_min_ns - disp_range_comp).clamp(min=0, max=max_disp)
        cur_disp_max = (disp_max_ns + disp_range_comp).clamp(min=0, max=max_disp)

        new_interval = (cur_disp_max - cur_disp_min) / (ndisp//4 - 1) #(B, 1/4H, 1/4W)

        # (B, 1/4D, 1/4H, 1/4W)
        disp_range_samples = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp//4, device=cur_disp.device,
                                                                      dtype=cur_disp.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1,
                                                                                                   1) * new_interval.unsqueeze(1))
        # (B, D, H, W)
        disp_range_samples = F.interpolate((disp_range_samples * 4.0).unsqueeze(1),
                                          [ndisp, H, W], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1)
    return disp_range_samples


def get_disp_range_samples(cur_disp, ndisp, disp_inteval_pixel, device, dtype, shape, using_ns, ns_size, max_disp=192.0):
    #shape, (B, H, W)
    #cur_disp: (B, H, W) or float
    #return disp_range_values: (B, D, H, W)
    # with torch.no_grad():
    if cur_disp is None:
        cur_disp = torch.tensor(0, device=device, dtype=dtype, requires_grad=False).reshape(1, 1, 1).repeat(*shape)
        cur_disp_min = (cur_disp - ndisp / 2 * disp_inteval_pixel).clamp(min=0.0)   #(B, H, W)
        cur_disp_max = (cur_disp_min + (ndisp - 1) * disp_inteval_pixel).clamp(max=max_disp)
        new_interval = (cur_disp_max - cur_disp_min) / (ndisp - 1)  # (B, H, W)

        disp_range_volume = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp, device=cur_disp.device,
                                                                      dtype=cur_disp.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))

    else:
        disp_range_volume = get_cur_disp_range_samples(cur_disp, ndisp, disp_inteval_pixel, shape, ns_size, using_ns, max_disp)

    return disp_range_volume


class res_submodule_attention(nn.Module):
    def __init__(self, scale, input_layer, value_planes=32, out_planes=64):
        super(res_submodule_attention, self).__init__()
        self.resample = resample2d
        self.pool = nn.AvgPool2d(2 ** scale, 2 ** scale)

        self.attention = SA_Module(input_nc=10)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer + 10, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes * 2),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_planes * 2, out_planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes * 4),
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(out_planes * 4, out_planes * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes * 4),
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(out_planes * 4, out_planes * 2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes * 2)
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(out_planes * 2, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.redir1 = nn.Sequential(
            nn.Conv2d(input_layer + 10, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes * 2, out_planes * 2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.res = nn.Conv2d(out_planes, 1, 1, 1, bias=False)

    def forward(self, left, right, disp, feature):
        scale = left.size()[2] / disp.size()[2]
        left = self.pool(left)
        right = self.pool(right)

        disp_ = disp / scale  # align the disparity to the proper scale

        left_rec = self.resample(right, disp_.squeeze(1))   #   new  resample
        error_map = left - left_rec

        query = torch.cat((left, right, error_map, disp_), dim=1)
        attention_map = self.attention(query)
        attented_feature = attention_map * torch.cat((feature, query), dim=1)

        conv1 = self.conv1(attented_feature)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(attented_feature), inplace=True)

        res = self.res(conv6) * scale
        # return res, attention_map, error_map
        return res


class SA_Module(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=16):
        super(SA_Module, self).__init__()
        self.attention_value = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, output_nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_value = self.attention_value(x)
        return attention_value

def resample2d(y, disp):
    
    if len(y.size())==4:
        bs, channels, height, width = y.size()
    else:
        bs,height,width=y.size()
        y=torch.unsqueeze(y,dim=1)
        channels=0


    mh, mw = torch.meshgrid([torch.arange(0, height, dtype=y.dtype, device=y.device),
                                 torch.arange(0, width, dtype=y.dtype, device=y.device)])  # (H *W)


    mh = mh.reshape(1, height, width).repeat(bs,  1, 1)
    mw = mw.reshape(1, height, width).repeat(bs,  1, 1)  # (B, H, W)

    cur_disp_coords_y = mh
    cur_disp_coords_x = mw - disp

    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
   # print(coords_x.shape,coords_y.shape)
    grid = torch.stack([coords_x, coords_y], dim=3) #(B,  H, W, 2)

    y_warped = F.grid_sample(y, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True).view([bs, channels, height, width] if channels!=0 else [bs,height,width])  #(B, C, H, W)

    return y_warped

if __name__=="__main__":
    x=torch.randint(10,[1,3,2,2])/255
    print(x)
    disp=torch.zeros([2,5,5])
    print(resample2d(x,disp))
