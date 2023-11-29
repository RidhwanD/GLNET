import torch
from torch import nn
import torchvision.models as models

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
        self.fc1 = nn.Linear(in_features=6*8*8, out_features=self.num_classes, bias=True)
        self.prelu = nn.PReLU()
        self.avgpool = nn.AvgPool2d(2)
        
        self.gradients_1 = None
        self.gradients_2 = None

    def make_back_bone(self,base_model):
        if base_model == 'vgg16':
            model = models.vgg16(weights='VGG16_Weights.DEFAULT')
            model.classifier[-1] =  nn.Linear(in_features=4096, out_features=self.num_classes, bias=True)
            weights = torch.load('saved_models/WHU-RS19_2023-11-29_vgg16_90.4_baseline.pth')           # Use saved parameters
            model.load_state_dict(weights)
            for param in model.parameters():
                if self.fixed:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            model.classifier[-1] = nn.Linear(in_features=4096,out_features=self.out_features_dim)
            return model

        if base_model == 'alexnet':
            model = models.alexnet(weights='AlexNet_Weights.DEFAULT')
            model.fc =  nn.Linear(in_features=4096, out_features=self.num_classes, bias=True)
            weights = torch.load('saved_models/WHU-RS19_2023-10-30_alexnet_92.4_baseline.pth')           # Use saved parameters
            model.load_state_dict(weights)
            for param in model.parameters():
                if self.fixed:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            model.classifier[-1] = nn.Linear(in_features=4096,out_features=self.out_features_dim)
            return model
        
        if base_model == 'resnet50':
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
            model.classifier[-1] =  nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)
            weights = torch.load('saved_models/WHU-RS19_2023-10-30_resnet50_88.8_baseline.pth')          # Use saved parameters
            model.load_state_dict(weights)
            for param in model.parameters():
                if self.fixed:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            model.fc = nn.Linear(in_features=2048,out_features=self.out_features_dim)

            return model
        
        
        
    # method for the activation extraction
    def get_activations(self, x):
        return self.lower_model(x)
    
    # hook for the gradients of the activations
    def activations_hook_image1(self, grad):
        self.gradients_1 = grad
    def activations_hook_image2(self, grad):
        self.gradients_2 = grad
        
    def get_activations_gradient(self, sub_network=1):
        if sub_network == 1:
            return self.gradients_1
        else:
            return self.gradients_2



    def forward(self, img, cluster_data):
        output_list = []
        for index,image in enumerate(cluster_data):
            output_list.append(self.upper_backbone(image))
            output_list[index] = torch.unsqueeze(output_list[index],1)

        x_upper = torch.cat((output_list),1).requires_grad_(True)
        _ = x_upper.register_hook(self.activations_hook_image2)
        x_upper = x_upper.view(x_upper.shape[0],5,16,16)

        x_lower = self.lower_model(img).requires_grad_(True)
        _ = x_lower.register_hook(self.activations_hook_image1)  
        x_lower = torch.unsqueeze(x_lower,1)
        x_lower = x_lower.view(x_lower.shape[0],1,16,16)

        x = torch.cat((x_upper,x_lower), dim = 1)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
