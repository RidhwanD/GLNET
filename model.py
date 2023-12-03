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
        
        self.gradients_lower = None
        self.gradients_upper_1 = None
        self.gradients_upper_2 = None
        self.gradients_upper_3 = None
        self.gradients_upper_4 = None
        self.gradients_upper_5 = None
        
    def make_back_bone(self,base_model):
        if base_model == 'vgg16':
            model = models.vgg16(weights='VGG16_Weights.DEFAULT')
            model.classifier[-1] =  nn.Linear(in_features=4096, out_features=self.num_classes, bias=True)
            weights = torch.load('saved_models/WHU-RS19_2023-12-02_vgg16_95.4_baseline.pth')           # Use saved parameters
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
            weights = torch.load('saved_models/WHU-RS19_2023-12-02_alexnet_84.6_baseline.pth')           # Use saved parameters
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
            weights = torch.load('saved_models/WHU-RS19_2023-12-02_resnet50_82.4_baseline.pth')          # Use saved parameters
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
    def get_activations_lower(self, x):
        return self.lower_model(x)
    def get_activations_upper(self, x):
        return self.upper_backbone(x)
    
    # hook for the gradients of the activations
    def activations_hook_image_lower(self, grad):
        self.gradients_lower = grad
    def activations_hook_image_upper_1(self, grad):
        self.gradients_upper_1 = grad
    def activations_hook_image_upper_2(self, grad):
        self.gradients_upper_2 = grad
    def activations_hook_image_upper_3(self, grad):
        self.gradients_upper_3 = grad
    def activations_hook_image_upper_4(self, grad):
        self.gradients_upper_4 = grad
    def activations_hook_image_upper_5(self, grad):
        self.gradients_upper_5 = grad
        
    def get_activations_gradient(self, sub_network=1, sub_input=None):
        if sub_network == 1:
            return self.gradients_lower
        else:
            if (sub_input):
                return self.gradients_upper[sub_input]
            else:
                return self.gradients_upper



    def forward(self, img, cluster_data):
        x_lower = self.lower_model(img).requires_grad_(True)
        _ = x_lower.register_hook(self.activations_hook_image_lower)
        if (self.base_model == 'alexnet'):
            x_lower = self.adaptive_pool(x_lower)
        else:
            x_lower = self.global_pool(x_lower)
        x_lower = x_lower.view(x_lower.size(0), -1)
        x_lower = self.fcn1(x_lower)
        x_lower = torch.unsqueeze(x_lower,1)
        x_lower = x_lower.view(x_lower.shape[0],1,16,16)
        
        
        
        output_list = []
        # for index,image in enumerate(cluster_data):
        output1 = self.upper_backbone(cluster_data[0]).requires_grad_(True)
        _ = output1.register_hook(lambda grad: self.activations_hook_image_upper_1(grad))
        if (self.base_model == 'alexnet'):
            output1 = self.adaptive_pool(output1)
        else:
            output1 = self.global_pool(output1)
        output1 = output1.view(output1.size(0), -1)
        output_list.append(torch.unsqueeze(self.fcn1(output1),1))

        output2 = self.upper_backbone(cluster_data[1]).requires_grad_(True)
        _ = output2.register_hook(lambda grad: self.activations_hook_image_upper_2(grad))
        if (self.base_model == 'alexnet'):
            output2 = self.adaptive_pool(output2)
        else:
            output2 = self.global_pool(output2)
        output2 = output2.view(output2.size(0), -1)
        output_list.append(torch.unsqueeze(self.fcn1(output2),1))

        output3 = self.upper_backbone(cluster_data[2]).requires_grad_(True)
        _ = output3.register_hook(lambda grad: self.activations_hook_image_upper_3(grad))
        if (self.base_model == 'alexnet'):
            output3 = self.adaptive_pool(output3)
        else:
            output3 = self.global_pool(output3)
        output3 = output3.view(output3.size(0), -1)
        output_list.append(torch.unsqueeze(self.fcn1(output3),1))

        output4 = self.upper_backbone(cluster_data[3]).requires_grad_(True)
        _ = output4.register_hook(lambda grad: self.activations_hook_image_upper_4(grad))
        if (self.base_model == 'alexnet'):
            output4 = self.adaptive_pool(output4)
        else:
            output4 = self.global_pool(output4)
        output4 = output4.view(output4.size(0), -1)
        output_list.append(torch.unsqueeze(self.fcn1(output4),1))

        output5 = self.upper_backbone(cluster_data[4]).requires_grad_(True)
        _ = output5.register_hook(lambda grad: self.activations_hook_image_upper_5(grad))
        if (self.base_model == 'alexnet'):
            output5 = self.adaptive_pool(output5)
        else:
            output5 = self.global_pool(output5)
        output5 = output5.view(output5.size(0), -1)
        output_list.append(torch.unsqueeze(self.fcn1(output5),1))
        
        

        x_upper = torch.cat((output_list),1) # .requires_grad_(True)
        # _ = x_upper.register_hook(self.activations_hook_image2)
        x_upper = x_upper.view(x_upper.shape[0],5,16,16)
        
        x = torch.cat((x_upper,x_lower), dim = 1)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        self.gradients_upper = [self.gradients_upper_1, self.gradients_upper_2, self.gradients_upper_3, self.gradients_upper_4, self.gradients_upper_5]

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
        
        self.gradients_lower = None
        self.gradients_upper_1 = None
        self.gradients_upper_2 = None
        self.gradients_upper_3 = None
        self.gradients_upper_4 = None
        self.gradients_upper_5 = None
        
        self.spp = SpatialPyramidPooling(levels=self.levels)
    
    
    def make_back_bone(self,base_model):
        if base_model == 'vgg16':
            model = models.vgg16(weights='VGG16_Weights.DEFAULT')
            model.classifier[-1] =  nn.Linear(in_features=4096, out_features=self.num_classes, bias=True)
            weights = torch.load('saved_models/WHU-RS19_2023-12-02_vgg16_95.4_baseline.pth')           # Use saved parameters
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
            weights = torch.load('saved_models/WHU-RS19_2023-12-02_alexnet_84.6_baseline.pth')           # Use saved parameters
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
            weights = torch.load('saved_models/WHU-RS19_2023-12-02_resnet50_82.4_baseline.pth')          # Use saved parameters
            model.load_state_dict(weights)
            for param in model.parameters():
                if self.fixed:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            self.fcn1 = nn.Linear(in_features=sum([(i*i)*2048 for i in self.levels]), out_features=self.out_features_dim)
            return nn.Sequential(*list(model.children())[:-2])
        
        
        
    # method for the activation extraction
    def get_activations_lower(self, x):
        return self.lower_model(x)
    def get_activations_upper(self, x):
        return self.upper_backbone(x)
    
    # hook for the gradients of the activations
    def activations_hook_image_lower(self, grad):
        self.gradients_lower = grad
    def activations_hook_image_upper_1(self, grad):
        self.gradients_upper_1 = grad
    def activations_hook_image_upper_2(self, grad):
        self.gradients_upper_2 = grad
    def activations_hook_image_upper_3(self, grad):
        self.gradients_upper_3 = grad
    def activations_hook_image_upper_4(self, grad):
        self.gradients_upper_4 = grad
    def activations_hook_image_upper_5(self, grad):
        self.gradients_upper_5 = grad
        
    def get_activations_gradient(self, sub_network=1, sub_input=None):
        if sub_network == 1:
            return self.gradients_lower
        else:
            if (sub_input):
                return self.gradients_upper[sub_input]
            else:
                return self.gradients_upper



    def forward(self, img, cluster_data):
        x_lower = self.lower_model(img).requires_grad_(True)
        _ = x_lower.register_hook(self.activations_hook_image_lower)
        x_lower = self.spp(x_lower)  # Apply SPP here
        x_lower = x_lower.view(x_lower.size(0), -1)
        x_lower = self.fcn1(x_lower)
        x_lower = torch.unsqueeze(x_lower,1)
        x_lower = x_lower.view(x_lower.shape[0],1,16,16)
        
        output_list = []
        output1 = self.upper_backbone(cluster_data[0]).requires_grad_(True)
        _ = output1.register_hook(lambda grad: self.activations_hook_image_upper_1(grad))
        output1 = self.spp(output1)  # Apply SPP here
        output1 = output1.view(output1.size(0), -1)
        output_list.append(self.fcn1(output1))
        output_list[0] = torch.unsqueeze(output_list[0],1)
        
        output2 = self.upper_backbone(cluster_data[1]).requires_grad_(True)
        _ = output2.register_hook(lambda grad: self.activations_hook_image_upper_2(grad))
        output2 = self.spp(output2)  # Apply SPP here
        output2 = output2.view(output2.size(0), -1)
        output_list.append(self.fcn1(output2))
        output_list[1] = torch.unsqueeze(output_list[1],1)
        
        output3 = self.upper_backbone(cluster_data[2]).requires_grad_(True)
        _ = output3.register_hook(lambda grad: self.activations_hook_image_upper_3(grad))
        output3 = self.spp(output3)  # Apply SPP here
        output3 = output3.view(output3.size(0), -1)
        output_list.append(self.fcn1(output3))
        output_list[2] = torch.unsqueeze(output_list[2],1)
        
        output4 = self.upper_backbone(cluster_data[3]).requires_grad_(True)
        _ = output4.register_hook(lambda grad: self.activations_hook_image_upper_4(grad))
        output4 = self.spp(output4)  # Apply SPP here
        output4 = output4.view(output4.size(0), -1)
        output_list.append(self.fcn1(output4))
        output_list[3] = torch.unsqueeze(output_list[3],1)
        
        output5 = self.upper_backbone(cluster_data[4]).requires_grad_(True)
        _ = output5.register_hook(lambda grad: self.activations_hook_image_upper_5(grad))
        output5 = self.spp(output5)  # Apply SPP here
        output5 = output5.view(output5.size(0), -1)
        output_list.append(self.fcn1(output5))
        output_list[4] = torch.unsqueeze(output_list[4],1)

        x_upper = torch.cat((output_list),1) # .requires_grad_(True)
        # _ = x_upper.register_hook(self.activations_hook_image2)
        x_upper = x_upper.view(x_upper.shape[0],5,16,16)
        
        x = torch.cat((x_upper,x_lower), dim = 1)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        self.gradients_upper = [self.gradients_upper_1, self.gradients_upper_2, self.gradients_upper_3, self.gradients_upper_4, self.gradients_upper_5]

        return x