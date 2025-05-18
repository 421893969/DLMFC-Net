import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


def my_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


print(111)


class fuse(nn.Module):
    def __init__(self):
        super(fuse, self).__init__()
        self.conv1_channel1 = nn.Conv2d(3, 3, 1, bias=True)
        self.conv1_spatial1 = nn.Conv2d(3, 1, 3, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2):
        rgb_att = F.sigmoid(self.conv1_spatial1(x1))
        thermal_att = F.sigmoid(self.conv1_spatial1(x2))
        rgb_att1 = rgb_att + rgb_att * thermal_att
        thermal_att = thermal_att + rgb_att * thermal_att
        spatial_attentioned_rgb_feat = thermal_att * x1
        spatial_attentioned_thermal_feat = rgb_att1 * x2

        rgb_vec = self.avg_pool(x1)
        rgb_vec = self.conv1_channel1(rgb_vec)
        rgb_vec = nn.Softmax(dim=1)(rgb_vec) * rgb_vec.shape[1]
        # img_vec = nn.Softmax(dim=1)(img_vec)
        thermal_vec = self.avg_pool(x2)
        thermal_vec = self.conv1_channel1(thermal_vec)
        thermal_vec = nn.Softmax(dim=1)(thermal_vec) * thermal_vec.shape[1]
        # rgb_vec = rgb_vec+rgb_vec*thermal_vec
        # thermal_vec = thermal_vec + rgb_vec * thermal_vec
        gray1 = spatial_attentioned_rgb_feat * thermal_vec
        gray2 = spatial_attentioned_thermal_feat * rgb_vec
        return gray1, gray2


class fuse2(nn.Module):
    def __init__(self):
        super(fuse2, self).__init__()
        # self.conv1_channel1 = nn.Conv2d(2048, 2048, 1, bias=True)
        self.conv1_spatial1 = nn.Conv2d(2048, 1, 3, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2):
        rgb_att = F.sigmoid(self.conv1_spatial1(x1))
        thermal_att = F.sigmoid(self.conv1_spatial1(x2))
        rgb_att1 = rgb_att + rgb_att * thermal_att
        thermal_att = thermal_att + rgb_att * thermal_att
        spatial_attentioned_rgb_feat = thermal_att * x1
        spatial_attentioned_thermal_feat = rgb_att1 * x2

        return spatial_attentioned_rgb_feat, spatial_attentioned_thermal_feat


class embed_net(nn.Module):
    def __init__(self, class_num, no_local='on', gm_pool='on', arch='resnet50'):
        super(embed_net, self).__init__()
        ##############################
        self.fuse = fuse()
        self.fuse2 = fuse2()
        ##############################

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck1 = nn.BatchNorm1d(pool_dim)
        self.bottleneck1.bias.requires_grad_(False)  # no shift
        self.bottleneck1.apply(weights_init_kaiming)
        self.classifier1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier1.apply(weights_init_classifier)

        self.bottleneck2 = nn.BatchNorm1d(pool_dim)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)
        self.classifier2 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier2.apply(weights_init_classifier)

        self.bottleneck3 = nn.BatchNorm1d(pool_dim)
        self.bottleneck3.bias.requires_grad_(False)  # no shift
        self.bottleneck3.apply(weights_init_kaiming)
        self.classifier3 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier3.apply(weights_init_classifier)

        self.bottleneck4 = nn.BatchNorm1d(pool_dim)
        self.bottleneck4.bias.requires_grad_(False)  # no shift
        self.bottleneck4.apply(weights_init_kaiming)
        self.classifier4 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier4.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.encode1 = nn.Conv2d(3, 1, 1)
        self.encode1.apply(my_weights_init)
        self.fc1 = nn.Conv2d(1, 1, 1)
        self.fc1.apply(my_weights_init)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn1.apply(weights_init_kaiming)

        self.encode2 = nn.Conv2d(3, 1, 1)
        self.encode2.apply(my_weights_init)
        self.fc2 = nn.Conv2d(1, 1, 1)
        self.fc2.apply(my_weights_init)
        self.bn2 = nn.BatchNorm2d(1)
        self.bn2.apply(weights_init_kaiming)

        self.decode = nn.Conv2d(1, 3, 1)
        self.decode.apply(my_weights_init)

        self.fc = nn.Linear(24 * 12, 1)
        self.softmax = nn.Softmax(dim=0)
        self.classifier = nn.Linear(pool_dim, class_num)
        self.w = nn.Parameter(torch.ones(2))
        self.w1 = nn.Parameter(torch.ones(2))

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            # 
            gray1, gray2 = self.fuse(x1, x2)
            # gray = (torch.cat((gray1, gray2), 0))
            # gray1, gray2 = torch.chunk(gray, 2, 0)
            # xo = torch.cat((x1, x2), 0)  # £¨64,3,384,192£©
            x1 = torch.cat((x1, gray1), 0)
            x1 = self.visible_module(x1)  # £¨64,64,96,48£©
            x2 = torch.cat((x2, gray2), 0)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)

            # x1 = self.visible_module(x1)
            # x2 = self.thermal_module(x2)
            # gray1,gray2 = self.fuse(x1,x2)
            # x1 = torch.cat((x1, gray1), 0)
            # x2 = torch.cat((x2, gray2),0)
            # x = torch.cat((x1, x2), 0)
            # (128,2048,24,12)


        elif modal == 1:
            # gray1 = F.relu(self.encode1(x1))  # self.encode1 = nn.Conv2d(3, 1, 1)
            # gray1 = self.bn1(F.relu(self.fc1(gray1)))
            # gray1 = F.relu(self.decode(gray1))  # self.decode = nn.Conv2d(1, 3, 1)
            x = self.visible_module(x1)
            # x = self.visible_module(torch.cat((x1, gray1), 0))

        elif modal == 2:
            # gray2 = F.relu(self.encode2(x2))
            # gray2 = self.bn2(F.relu(self.fc2(gray2)))
            # gray2 = F.relu(self.decode(gray2))
            x = self.thermal_module(x2)
            # x = self.thermal_module(torch.cat((x2, gray2), 0))
        x = self.base_resnet.base.layer1(x)
        x = self.base_resnet.base.layer2(x)
        x = self.base_resnet.base.layer3(x)
        x = self.base_resnet.base.layer4(x)
        # shared block

        if modal == 0:
            x11, x12, x21, x22 = torch.chunk(x, 4, 0)
            x111, x211 = self.fuse2(x11, x21)
            x12 = x12 + x111
            x22 = x211 + x22
            x = torch.cat((x11, x12, x21, x22), 0)

            b, c, h, w = x11.size()
            x11 = x11.view(b, c, -1)
            x12 = x12.view(b, c, -1)
            x21 = x21.view(b, c, -1)
            x22 = x22.view(b, c, -1)
            x11 = self.fc(x11).view(b, c)
            p = 3.0  # regDB: 10.0    SYSU: 3.0
            x12 = (torch.mean(x12 ** p, dim=-1) + 1e-12) ** (1 / p)
            x12 = self.softmax(x12)
            w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
            w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
            w3 = torch.exp(self.w1[0]) / torch.sum(torch.exp(self.w1))
            w4 = torch.exp(self.w1[1]) / torch.sum(torch.exp(self.w1))
            x11 = w1 * x11 + w2 * x12

            x21 = self.fc(x21).view(b, c)
            x22 = (torch.mean(x22 ** p, dim=-1) + 1e-12) ** (1 / p)
            x22 = self.softmax(x22)
            x21 = w3 * x21 + w4 * x22
        # 
        #     #
        x41, x42, x43, x44 = torch.chunk(x, 4, 2)
        # x00 = self.avgpool(x)
        # x00 = x00.view(x00.size(0), x00.size(1))
        # x011,x022,x033,x044 = torch.chunk(x00,4,0)

        # x011 = self.classifier(x011)
        # x022 = self.classifier(x022)
        # x033 = self.classifier(x033)
        # x044 = self.classifier(x044)

        x41 = self.avgpool(x41)
        x42 = self.avgpool(x42)
        x43 = self.avgpool(x43)
        x44 = self.avgpool(x44)
        x41 = x41.view(x41.size(0), x41.size(1))
        x42 = x42.view(x42.size(0), x42.size(1))
        x43 = x43.view(x43.size(0), x43.size(1))
        x44 = x44.view(x44.size(0), x44.size(1))
        # 
        feat41 = self.bottleneck1(x41)
        feat42 = self.bottleneck2(x42)
        feat43 = self.bottleneck3(x43)
        feat44 = self.bottleneck4(x44)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), x.size(1))
        # feat = self.bottleneck1(x)
        
        if self.training:
            return x41, x42, x43, x44, self.classifier1(feat41), self.classifier2(feat42), self.classifier3(
                feat43), self.classifier4(feat44),x11,x21,self.classifier(x11),self.classifier(x21)
        else:
            return self.l2norm(torch.cat((x41, x42, x43, x44), 1)), self.l2norm(
                torch.cat((feat41, feat42, feat43, feat44), 1))
