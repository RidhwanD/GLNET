import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

torch.nn.Module.dump_patches = True
class SiameseNetwork(nn.Module):
    def __init__(self,base_model ='vgg16',num_classes = 5 , fixed = False, out_features_dim = 256):
        super(SiameseNetwork, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.fixed = fixed
        self.out_features_dim = out_features_dim

        self.lower_model = self.make_back_bone(self.base_model)
        self.upper_backbone = self.make_back_bone(self.base_model)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(in_features=6*8*8, out_features=self.num_classes, bias=True)
        self.prelu = nn.PReLU()
        self.avgpool = nn.AvgPool2d(2)
        
        self.gradients_1 = None
        self.gradients_2 = [None, None, None, None, None]

    def make_back_bone(self,base_model):
        if base_model == 'vgg16':
            model = models.vgg16(weights='VGG16_Weights.DEFAULT')
            model.classifier[-1] =  nn.Linear(in_features=4096, out_features=self.num_classes, bias=True)
            weights = torch.load('saved_models/rsscn7_2023-11-30_vgg16_89.79_baseline.pth')           # Use saved parameters
            model.load_state_dict(weights)
            for param in model.parameters():
                if self.fixed:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            # model.classifier[-1] = nn.Linear(in_features=4096,out_features=self.out_features_dim)
            self.fcn1 = nn.Linear(in_features=512,out_features=self.out_features_dim)
            return model.features

        if base_model == 'alexnet':
            model = models.alexnet(weights='AlexNet_Weights.DEFAULT')
            model.classifier[-1] =  nn.Linear(in_features=4096, out_features=self.num_classes, bias=True)
            weights = torch.load('saved_models/rsscn7_2023-11-30_alexnet_87.07_baseline.pth')           # Use saved parameters
            model.load_state_dict(weights)
            for param in model.parameters():
                if self.fixed:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            model.classifier[-1] = nn.Linear(in_features=4096,out_features=self.out_features_dim)
            self.fcn1 = model.classifier
            return model.features
        
        if base_model == 'resnet50':
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
            model.fc =  nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)
            weights = torch.load('saved_models/rsscn7_2023-11-30_resnet50_86.93_baseline.pth')          # Use saved parameters
            model.load_state_dict(weights)
            for param in model.parameters():
                if self.fixed:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            # model.fc = nn.Linear(in_features=2048,out_features=self.out_features_dim)
            self.fcn1 = nn.Linear(in_features=2048,out_features=self.out_features_dim)
            return nn.Sequential(*list(model.children())[:-2])
        
        
        
    # method for the activation extraction
    def get_activations(self, x):
        return self.lower_model(x)
    
    # hook for the gradients of the activations
    def activations_hook_image1(self, grad):
        self.gradients_1 = grad
    def activations_hook_image2(self, grad, index):
        self.gradients_2[index] = grad
        
    def get_activations_gradient(self, sub_network=1, sub_input=None):
        if sub_network == 1:
            return self.gradients_1
        else:
            if (sub_input):
                return self.gradients_2[sub_input]
            else:
                return self.gradients_2



    def forward(self, img, cluster_data):
        x_lower = self.lower_model(img).requires_grad_(True)
        _ = x_lower.register_hook(self.activations_hook_image1)
        if (self.base_model == 'alexnet'):
            x_lower = self.adaptive_pool(x_lower)
            x_lower = x_lower.view(x_lower.size(0), -1)
        else:
            x_lower = self.global_pool(x_lower)
            x_lower = x_lower.view(x_lower.size(0), -1)
        x_lower = self.fcn1(x_lower)
        x_lower = torch.unsqueeze(x_lower,1)
        x_lower = x_lower.view(x_lower.shape[0],1,16,16)
        
        output_list = []
        for index,image in enumerate(cluster_data):
            output = self.upper_backbone(image).requires_grad_(True)
            _ = output.register_hook(lambda grad: self.activations_hook_image2(grad, index))
            if (self.base_model == 'alexnet'):
                output = self.adaptive_pool(output)
                output = output.view(output.size(0), -1)
            else:
                output = self.global_pool(output)
                output = output.view(output.size(0), -1)
            output_list.append(self.fcn1(output))
            output_list[index] = torch.unsqueeze(output_list[index],1)

        x_upper = torch.cat((output_list),1) # .requires_grad_(True)
        # _ = x_upper.register_hook(self.activations_hook_image2)
        x_upper = x_upper.view(x_upper.shape[0],5,16,16)
        
        x = torch.cat((x_upper,x_lower), dim = 1)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels):
        super(SpatialPyramidPooling, self).__init__()
        self.levels = levels

    def forward(self, x):
        n, c, h, w = x.size()
        for i in range(len(self.levels)):
            level = self.levels[i]
            # kernel_size = (h // level, w // level)
            # stride = kernel_size
            pooling = F.adaptive_avg_pool2d(x, output_size=level)
            if i == 0:
                spp = pooling.view(n, -1)
            else:
                spp = torch.cat((spp, pooling.view(n, -1)), 1)
        return spp

class SiameseNetworkWithSPP(nn.Module):
    def __init__(self,base_model ='vgg16',num_classes = 5 , fixed = False, out_features_dim = 256):
        super(SiameseNetworkWithSPP, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.fixed = fixed
        self.out_features_dim = out_features_dim
        self.levels = [1, 2, 4]
        
        self.lower_model = self.make_back_bone(self.base_model)
        self.upper_backbone = self.make_back_bone(self.base_model)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(in_features=6*8*8, out_features=self.num_classes, bias=True)
        self.prelu = nn.PReLU()
        self.avgpool = nn.AvgPool2d(2)
        
        self.gradients_1 = None
        self.gradients_2 = [None, None, None, None, None]
        
        self.spp = SpatialPyramidPooling(levels=self.levels)
    
    
    def make_back_bone(self,base_model):
        if base_model == 'vgg16':
            model = models.vgg16(weights='VGG16_Weights.DEFAULT')
            model.classifier[-1] =  nn.Linear(in_features=4096, out_features=self.num_classes, bias=True)
            weights = torch.load('saved_models/rsscn7_2023-11-30_vgg16_89.79_baseline.pth')           # Use saved parameters
            model.load_state_dict(weights)
            for param in model.parameters():
                if self.fixed:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            self.fcn1 = nn.Linear(in_features=sum([(i*i)*512 for i in self.levels]), out_features=self.out_features_dim)
            return model.features

        if base_model == 'alexnet':
            model = models.alexnet(weights='AlexNet_Weights.DEFAULT')
            model.classifier[-1] =  nn.Linear(in_features=4096, out_features=self.num_classes, bias=True)
            weights = torch.load('saved_models/rsscn7_2023-11-30_alexnet_87.07_baseline.pth')           # Use saved parameters
            model.load_state_dict(weights)
            for param in model.parameters():
                if self.fixed:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            model.classifier[-1] = nn.Linear(in_features=4096,out_features=self.out_features_dim)
            self.fcn1 = nn.Linear(in_features=sum([(i*i)*256 for i in self.levels]), out_features=self.out_features_dim)
            return model.features
        
        if base_model == 'resnet50':
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
            model.fc =  nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)
            weights = torch.load('saved_models/rsscn7_2023-11-30_resnet50_86.93_baseline.pth')          # Use saved parameters
            model.load_state_dict(weights)
            for param in model.parameters():
                if self.fixed:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            self.fcn1 = nn.Linear(in_features=sum([(i*i)*2048 for i in self.levels]), out_features=self.out_features_dim)
            return nn.Sequential(*list(model.children())[:-2])
        
        
        
    # method for the activation extraction
    def get_activations(self, x):
        return self.lower_model(x)
    
    # hook for the gradients of the activations
    def activations_hook_image1(self, grad):
        self.gradients_1 = grad
    def activations_hook_image2(self, grad, index):
        self.gradients_2[index] = grad
        
    def get_activations_gradient(self, sub_network=1, sub_input=None):
        if sub_network == 1:
            return self.gradients_1
        else:
            if (sub_input):
                return self.gradients_2[sub_input]
            else:
                return self.gradients_2



    def forward(self, img, cluster_data):
        x_lower = self.lower_model(img).requires_grad_(True)
        _ = x_lower.register_hook(self.activations_hook_image1)
        x_lower = self.spp(x_lower)  # Apply SPP here
        x_lower = x_lower.view(x_lower.size(0), -1)
        x_lower = self.fcn1(x_lower)
        x_lower = torch.unsqueeze(x_lower,1)
        x_lower = x_lower.view(x_lower.shape[0],1,16,16)
        
        output_list = []
        for index,image in enumerate(cluster_data):
            output = self.upper_backbone(image).requires_grad_(True)
            _ = output.register_hook(lambda grad: self.activations_hook_image2(grad, index))
            output = self.spp(output)  # Apply SPP here
            output = output.view(output.size(0), -1)
            output_list.append(self.fcn1(output))
            output_list[index] = torch.unsqueeze(output_list[index],1)

        x_upper = torch.cat((output_list),1) # .requires_grad_(True)
        # _ = x_upper.register_hook(self.activations_hook_image2)
        x_upper = x_upper.view(x_upper.shape[0],5,16,16)
        
        x = torch.cat((x_upper,x_lower), dim = 1)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x