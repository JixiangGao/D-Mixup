import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import json
from efficientnet_pytorch import EfficientNet

    
class Resnet50(nn.Module):
    def __init__(self, d=1024, out_size=383):
        super(Resnet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.region_wise_layers = nn.Sequential(
            nn.Linear(2048, d),
            nn.Tanh(),
            nn.Linear(d, out_size),
            nn.Sigmoid()
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = False
    
    def forward(self, ipt):
        x = self.resnet50.conv1(ipt)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        
        x = F.dropout(x, training=self.dropout)
        
        fi = x.permute(0, 2, 3, 1)                                 # bs x 7 x 7 x 2048
        P_ingre_i = self.region_wise_layers(fi)                    # bs x 7 x 7 x 383
        P_ingre_I = self.max(P_ingre_i.permute(0, 3, 1, 2)).view(ipt.size(0), -1)  # bs x 383
        
        # fi = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))   # bs x 2048 x 49
        # fi = fi.permute(0, 2, 1)                                 # bs x 49 x 2048
        # P_ingre_i = self.region_wise_layers(fi)                  # bs x 49 x 383
        # P_ingre_I, _ = torch.max(P_ingre_i, dim=1)               # bs x 383
        
        return P_ingre_I


class Resnet50_OneFC(nn.Module):
    def __init__(self, d=1024, out_size=383):
        super(Resnet50_OneFC, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.region_wise_layers = nn.Sequential(
            nn.Linear(2048, out_size),
            nn.Softmax(dim=3)
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = False
    
    def forward(self, ipt):
        x = self.resnet50.conv1(ipt)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        
        x = F.dropout(x, training=self.dropout)
        
        fi = x.permute(0, 2, 3, 1)                                 # bs x 7 x 7 x 2048
        P_ingre_i = self.region_wise_layers(fi)                    # bs x 7 x 7 x 383
        P_ingre_I = self.max(P_ingre_i.permute(0, 3, 1, 2)).view(ipt.size(0), -1)  # bs x 383
        
        # fi = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))   # bs x 2048 x 49
        # fi = fi.permute(0, 2, 1)                                 # bs x 49 x 2048
        # P_ingre_i = self.region_wise_layers(fi)                  # bs x 49 x 383
        # P_ingre_I, _ = torch.max(P_ingre_i, dim=1)               # bs x 383
        
        return P_ingre_I


class Resnet50_OneFC_Sigmoid(nn.Module):
    def __init__(self, d=1024, out_size=383):
        super(Resnet50_OneFC_Sigmoid, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.region_wise_layers = nn.Sequential(
            nn.Linear(2048, out_size),
            nn.Sigmoid()
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = False
    
    def forward(self, ipt):
        x = self.resnet50.conv1(ipt)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        
        x = F.dropout(x, training=self.dropout)
        
        fi = x.permute(0, 2, 3, 1)                                 # bs x 7 x 7 x 2048
        P_ingre_i = self.region_wise_layers(fi)                    # bs x 7 x 7 x 383
        P_ingre_I = self.max(P_ingre_i.permute(0, 3, 1, 2)).view(ipt.size(0), -1)  # bs x 383
        
        # fi = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))   # bs x 2048 x 49
        # fi = fi.permute(0, 2, 1)                                 # bs x 49 x 2048
        # P_ingre_i = self.region_wise_layers(fi)                  # bs x 49 x 383
        # P_ingre_I, _ = torch.max(P_ingre_i, dim=1)               # bs x 383
        
        return P_ingre_I


class Resnet101_OneFC(nn.Module):
    def __init__(self, d=1024, out_size=383):
        super(Resnet101_OneFC, self).__init__()
        self.resnet101 = models.resnet101(pretrained=True)
        self.region_wise_layers = nn.Sequential(
            nn.Linear(2048, out_size),
            nn.Softmax(dim=3)
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = False
    
    def forward(self, ipt):
        x = self.resnet101.conv1(ipt)
        x = self.resnet101.bn1(x)
        x = self.resnet101.relu(x)
        x = self.resnet101.maxpool(x)

        x = self.resnet101.layer1(x)
        x = self.resnet101.layer2(x)
        x = self.resnet101.layer3(x)
        x = self.resnet101.layer4(x)
        
        x = F.dropout(x, training=self.dropout)
        
        fi = x.permute(0, 2, 3, 1)                                 # bs x 7 x 7 x 2048
        P_ingre_i = self.region_wise_layers(fi)                    # bs x 7 x 7 x 383
        P_ingre_I = self.max(P_ingre_i.permute(0, 3, 1, 2)).view(ipt.size(0), -1)  # bs x 383
        
        # fi = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))   # bs x 2048 x 49
        # fi = fi.permute(0, 2, 1)                                 # bs x 49 x 2048
        # P_ingre_i = self.region_wise_layers(fi)                  # bs x 49 x 383
        # P_ingre_I, _ = torch.max(P_ingre_i, dim=1)               # bs x 383
        
        return P_ingre_I


class Resnet101_OneFC_Sigmoid(nn.Module):
    def __init__(self, d=1024, out_size=383):
        super(Resnet101_OneFC_Sigmoid, self).__init__()
        self.resnet101 = models.resnet101(pretrained=True)
        self.region_wise_layers = nn.Sequential(
            nn.Linear(2048, out_size),
            nn.Sigmoid()
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = False
    
    def forward(self, ipt):
        x = self.resnet101.conv1(ipt)
        x = self.resnet101.bn1(x)
        x = self.resnet101.relu(x)
        x = self.resnet101.maxpool(x)

        x = self.resnet101.layer1(x)
        x = self.resnet101.layer2(x)
        x = self.resnet101.layer3(x)
        x = self.resnet101.layer4(x)
        
        x = F.dropout(x, training=self.dropout)
        
        fi = x.permute(0, 2, 3, 1)                                 # bs x 7 x 7 x 2048
        P_ingre_i = self.region_wise_layers(fi)                    # bs x 7 x 7 x 383
        P_ingre_I = self.max(P_ingre_i.permute(0, 3, 1, 2)).view(ipt.size(0), -1)  # bs x 383
        
        # fi = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))   # bs x 2048 x 49
        # fi = fi.permute(0, 2, 1)                                 # bs x 49 x 2048
        # P_ingre_i = self.region_wise_layers(fi)                  # bs x 49 x 383
        # P_ingre_I, _ = torch.max(P_ingre_i, dim=1)               # bs x 383
        
        return P_ingre_I


class VGG19(nn.Module):
    def __init__(self, d=512, out_size=383):
        super(VGG19, self).__init__()
        net = models.vgg19(pretrained=True)
        self.features = net.features
        self.region_wise_layers = nn.Sequential(
            nn.Linear(512, d),
            nn.Tanh(),
            nn.Linear(d, out_size),
            nn.Sigmoid()
            # nn.Softmax(dim=3)
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = False

    def forward(self, ipt):
        x = self.features(ipt)
        x = F.dropout(x, training=self.dropout)
        
        fi = x.permute(0, 2, 3, 1)                                 # bs x 7 x 7 x d
        P_ingre_i = self.region_wise_layers(fi)                    # bs x 7 x 7 x 383
        P_ingre_I = self.max(P_ingre_i.permute(0, 3, 1, 2)).view(ipt.size(0), -1)  # bs x 383
        return P_ingre_I


class VGG19_Softmax(nn.Module):
    def __init__(self, d=512, out_size=383):
        super(VGG19_Softmax, self).__init__()
        net = models.vgg19(pretrained=True)
        self.features = net.features
        self.region_wise_layers = nn.Sequential(
            nn.Linear(512, d),
            nn.Tanh(),
            nn.Linear(d, out_size),
            nn.Softmax(dim=3)
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = False

    def forward(self, ipt):
        x = self.features(ipt)
        # print(x.shape)
        x = F.dropout(x, training=self.dropout)
        
        fi = x.permute(0, 2, 3, 1)                                 # bs x 7 x 7 x 512
        P_ingre_i = self.region_wise_layers(fi)                    # bs x 7 x 7 x 383
        # print(P_ingre_i.shape)
        P_ingre_I = self.max(P_ingre_i.permute(0, 3, 1, 2)).view(ipt.size(0), -1)  # bs x 383
        # print(P_ingre_I.shape)
        # exit(0)
        return P_ingre_I


class VGG19_global_sigmoid(nn.Module):
    def __init__(self, d=512, out_size=383):
        super(VGG19_global_sigmoid, self).__init__()
        net = models.vgg19(pretrained=True)
        self.features = net.features
        # self.region_wise_layers = nn.Sequential(
        #     nn.Linear(512, d),
        #     nn.Tanh(),
        #     nn.Linear(d, out_size),
        #     nn.Sigmoid()
        #     # nn.Softmax(dim=3)
        # )
        # self.max = nn.AdaptiveMaxPool2d((1, 1))
        # self.dropout = False
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.features(ipt)
        # x = F.dropout(x, training=self.dropout)
        
        # fi = x.permute(0, 2, 3, 1)                                 # bs x 7 x 7 x d
        # P_ingre_i = self.region_wise_layers(fi)                    # bs x 7 x 7 x 383
        # P_ingre_I = self.max(P_ingre_i.permute(0, 3, 1, 2)).view(ipt.size(0), -1)  # bs x 383
        # return P_ingre_I
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Resnet50_global_sigmoid(nn.Module):
    def __init__(self, d=2048, out_size=383):
        super(Resnet50_global_sigmoid, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.region_wise_layers = nn.Sequential(
            nn.Linear(2048, out_size),
            # nn.Tanh(),
            # nn.Linear(d, out_size),
            nn.Sigmoid()
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = False

    
    def forward(self, ipt):
        x = self.resnet50.conv1(ipt)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        
        x = F.dropout(x, training=self.dropout)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        P = self.region_wise_layers(x)
        return P


class Resnet101_global_sigmoid(nn.Module):
    def __init__(self, d=2048, out_size=383):
        super(Resnet101_global_sigmoid, self).__init__()
        self.resnet101 = models.resnet101(pretrained=True)
        self.region_wise_layers = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Linear(d, out_size),
            nn.Sigmoid()
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, ipt):
        x = self.resnet101.conv1(ipt)
        x = self.resnet101.bn1(x)
        x = self.resnet101.relu(x)
        x = self.resnet101.maxpool(x)

        x = self.resnet101.layer1(x)
        x = self.resnet101.layer2(x)
        x = self.resnet101.layer3(x)
        x = self.resnet101.layer4(x)
        
        x = F.dropout(x, training=self.dropout)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        P = self.region_wise_layers(x)
        return P


class DenseNet121_sigmoid(nn.Module):
    def __init__(self, d=2048, out_size=383):
        super(DenseNet121_sigmoid, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        self.region_wise_layers = nn.Sequential(
            nn.Linear(1024, out_size),
            nn.Sigmoid()
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = False
    
    def forward(self, ipt):
        features = self.densenet121.features(ipt)
        out = F.relu(features, inplace=True)

        fi = out.permute(0, 2, 3, 1)                                 # bs x 7 x 7 x 2048
        P_ingre_i = self.region_wise_layers(fi)                    # bs x 7 x 7 x 383
        P_ingre_I = self.max(P_ingre_i.permute(0, 3, 1, 2)).view(ipt.size(0), -1)  # bs x 383
        return P_ingre_I


class DenseNet121_softmax(nn.Module):
    def __init__(self, d=2048, out_size=383):
        super(DenseNet121_softmax, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        self.region_wise_layers = nn.Sequential(
            nn.Linear(1024, out_size),
            nn.Softmax(dim=3)
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = False
    
    def forward(self, ipt):
        features = self.densenet121.features(ipt)
        out = F.relu(features, inplace=True)

        fi = out.permute(0, 2, 3, 1)                                 # bs x 7 x 7 x 2048
        P_ingre_i = self.region_wise_layers(fi)                    # bs x 7 x 7 x 383
        P_ingre_I = self.max(P_ingre_i.permute(0, 3, 1, 2)).view(ipt.size(0), -1)  # bs x 383
        return P_ingre_I


class DenseNet121_global_sigmoid(nn.Module):
    def __init__(self, d=2048, out_size=383):
        super(DenseNet121_global_sigmoid, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        self.region_wise_layers = nn.Sequential(
            nn.Linear(1024, out_size),
            nn.Sigmoid()
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, ipt):
        features = self.densenet121.features(ipt)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        P = self.region_wise_layers(out)
        return P


class EfficientNetB7_global_sigmoid(nn.Module):
    def __init__(self, d=2048, out_size=383):
        super(EfficientNetB7_global_sigmoid, self).__init__()
        self.efficientnet_b7 = EfficientNet.from_pretrained('efficientnet-b7')
        self.region_wise_layers = nn.Sequential(
            nn.Linear(2560, out_size),
            nn.Sigmoid()
        )
    
    def forward(self, ipt):
        x = self.efficientnet_b7.extract_features(ipt)

        x = self.efficientnet_b7._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.efficientnet_b7._dropout(x)
        x = self.region_wise_layers(x)
        return x


import torch.hub
class SENet_res50_global_sigmoid(nn.Module):
    def __init__(self, d=2048, out_size=383):
        super(SENet_res50_global_sigmoid, self).__init__()
        self.senet_model = torch.hub.load(
            'moskomule/senet.pytorch',
            'se_resnet50',
            pretrained=True,)
        self.region_wise_layers = nn.Sequential(
            nn.Linear(2048, out_size),
            # nn.Tanh(),
            # nn.Linear(d, out_size),
            nn.Sigmoid()
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, ipt):
        x = self.senet_model.conv1(ipt)
        x = self.senet_model.bn1(x)
        x = self.senet_model.relu(x)
        x = self.senet_model.maxpool(x)

        x = self.senet_model.layer1(x)
        x = self.senet_model.layer2(x)
        x = self.senet_model.layer3(x)
        x = self.senet_model.layer4(x)
        
        x = F.dropout(x, training=self.dropout)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        P = self.region_wise_layers(x)
        return P


class SENet_res50_sigmoid(nn.Module):
    def __init__(self, d=1024, out_size=383):
        super(SENet_res50_sigmoid, self).__init__()
        self.senet_model = torch.hub.load(
            'moskomule/senet.pytorch',
            'se_resnet50',
            pretrained=True,)
        self.region_wise_layers = nn.Sequential(
            nn.Linear(2048, out_size),
            nn.Sigmoid()
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = False
    
    def forward(self, ipt):
        x = self.senet_model.conv1(ipt)
        x = self.senet_model.bn1(x)
        x = self.senet_model.relu(x)
        x = self.senet_model.maxpool(x)

        x = self.senet_model.layer1(x)
        x = self.senet_model.layer2(x)
        x = self.senet_model.layer3(x)
        x = self.senet_model.layer4(x)
        
        
        x = F.dropout(x, training=self.dropout)
        
        fi = x.permute(0, 2, 3, 1)                                 # bs x 7 x 7 x 2048
        P_ingre_i = self.region_wise_layers(fi)                    # bs x 7 x 7 x 383
        P_ingre_I = self.max(P_ingre_i.permute(0, 3, 1, 2)).view(ipt.size(0), -1)  # bs x 383
        
        return P_ingre_I


class SENet_res50_softmax(nn.Module):
    def __init__(self, d=1024, out_size=383):
        super(SENet_res50_softmax, self).__init__()
        self.senet_model = torch.hub.load(
            'moskomule/senet.pytorch',
            'se_resnet50',
            pretrained=True,)
        self.region_wise_layers = nn.Sequential(
            nn.Linear(2048, out_size),
            nn.Softmax(dim=3)
        )
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = False
    
    def forward(self, ipt):
        x = self.senet_model.conv1(ipt)
        x = self.senet_model.bn1(x)
        x = self.senet_model.relu(x)
        x = self.senet_model.maxpool(x)

        x = self.senet_model.layer1(x)
        x = self.senet_model.layer2(x)
        x = self.senet_model.layer3(x)
        x = self.senet_model.layer4(x)
        
        x = F.dropout(x, training=self.dropout)
        
        fi = x.permute(0, 2, 3, 1)                                 # bs x 7 x 7 x 2048
        P_ingre_i = self.region_wise_layers(fi)                    # bs x 7 x 7 x 383
        P_ingre_I = self.max(P_ingre_i.permute(0, 3, 1, 2)).view(ipt.size(0), -1)  # bs x 383
        
        return P_ingre_I


if __name__ == '__main__':
    VGG19()
