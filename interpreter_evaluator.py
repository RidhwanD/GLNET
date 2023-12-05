import numpy as np
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.nn.functional as F
from utils import  generateCluster
from gradcam_glnet import convert_to_tensor, Grad_CAM_GLNet, Grad_CAMpp_GLNet, Score_CAM_GLNet, Smooth_Score_CAM_GLNet, Smooth_Grad_CAMpp_GLNet, Median_Grad_CAM_GLNet
from gradcam import Grad_CAM, Grad_CAMpp, Score_CAM, Smooth_Score_CAM, Smooth_Grad_CAMpp, Median_Grad_CAM

def evaluate_exp(images, transform, model, explainability_method, device, partition):
    model.eval()  # Set the model to evaluation mode
    average_drop = []
    increases_in_confidence = []
    for img, target_class in tqdm(images):
        img = np.array(img)
        img1 = convert_to_tensor(img, transform, device).unsqueeze(0)
        image, cluster = generateCluster(img, transform)
        image = image.to(device).unsqueeze(0)
        cluster =  [item.to(device).unsqueeze(0) for item in cluster ]

        original_confidence = F.softmax(model(image, cluster), dim=1)[0]

        if explainability_method == Grad_CAMpp_GLNet:
            important_regions = explainability_method(img1, cluster, partition, model, img[:,:,0].shape)
        elif explainability_method == Median_Grad_CAM_GLNet:
            important_regions = explainability_method(img1, cluster, partition, model, img[:,:,0].shape)
        elif explainability_method == Score_CAM_GLNet:
            important_regions = explainability_method(img1, cluster, model, target_class, partition, device, img[:,:,0].shape)
        elif explainability_method == Smooth_Score_CAM_GLNet:
            important_regions = explainability_method(img1, cluster, model, target_class, partition, device, img[:,:,0].shape)
        elif explainability_method == Smooth_Grad_CAMpp_GLNet:
            important_regions = explainability_method(img1, cluster, model, partition, device, img[:,:,0].shape)
        elif explainability_method == Grad_CAMpp:
            important_regions = explainability_method(img1, cluster, model, img[:,:,0].shape)
        elif explainability_method == Median_Grad_CAM:
            important_regions = explainability_method(img1, cluster, model, img[:,:,0].shape)
        elif explainability_method == Score_CAM:
            important_regions = explainability_method(img1, cluster, model, target_class, device, img[:,:,0].shape)
        elif explainability_method == Smooth_Score_CAM:
            important_regions = explainability_method(img1, cluster, model, target_class, device, img[:,:,0].shape)
        elif explainability_method == Smooth_Grad_CAMpp:
            important_regions = explainability_method(img1, cluster, model, device, img[:,:,0].shape)
        
        important_regions = torch.from_numpy(important_regions).to(device)
        img = torch.from_numpy(img).to(device)
        occluded_image = occlude_regions(img, important_regions).detach().cpu().numpy()
        ni_occluded_image = occlude_non_important_regions(img, important_regions).detach().cpu().numpy()
        # plt.imshow(occluded_image)
        
        occluded_image, occluded_cluster = generateCluster(occluded_image, transform)
        occluded_image = occluded_image.to(device).unsqueeze(0)
        occluded_cluster =  [item.to(device).unsqueeze(0) for item in occluded_cluster ]
        
        occluded_confidence = F.softmax(model(occluded_image, occluded_cluster), dim=1)[0]
        
        ni_occluded_image, ni_occluded_cluster = generateCluster(ni_occluded_image, transform)
        ni_occluded_image = ni_occluded_image.to(device).unsqueeze(0)
        ni_occluded_cluster =  [item.to(device).unsqueeze(0) for item in ni_occluded_cluster ]
        
        ni_occluded_confidence = F.softmax(model(ni_occluded_image, ni_occluded_cluster), dim=1)[0]
        
        # Calculate metrics
        drop = (original_confidence[target_class] - occluded_confidence[target_class]) / original_confidence[target_class]
        average_drop.append(drop.detach().cpu().numpy())  # Move to CPU and convert to NumPy array if needed
        # print(i, drop)
        
        increase = (ni_occluded_confidence[target_class] - original_confidence[target_class]) / original_confidence[target_class]
        # print(i, increase)
        if increase > 0:
            increases_in_confidence.append(1)
        else:
            increases_in_confidence.append(0)
        
    # Aggregate results
    avg_drop_percentage = np.mean(average_drop) * 100
    percent_increase = (sum(increases_in_confidence) / len(images)) * 100

    return avg_drop_percentage, percent_increase

def heatmap_to_binary_mask(heatmap, threshold=0.5):
    return (heatmap > threshold).type(torch.uint8)

def occlude_regions(image_tensor, occlusion_heatmap):
    """
    Occludes the regions of the image specified by the occlusion heatmap.

    Parameters:
    image_tensor (Tensor): The image tensor (H x W x C).
    occlusion_heatmap (Tensor): A heatmap (H x W) indicating occlusion intensity.

    Returns:
    Tensor: The occluded image.
    """

    # Reshape and replicate the heatmap to match the image dimensions (H x W x C)
    expanded_heatmap = occlusion_heatmap[:, :, None]
    expanded_heatmap = expanded_heatmap.repeat(1, 1, image_tensor.shape[2])
    expanded_heatmap = heatmap_to_binary_mask(expanded_heatmap, torch.mean(expanded_heatmap.float()))

    # Apply the occlusion
    return image_tensor * (1 - expanded_heatmap)

def occlude_non_important_regions(image_tensor, occlusion_heatmap):
    """
    Occludes the regions of the image specified by the occlusion heatmap.

    Parameters:
    image_tensor (Tensor): The image tensor (H x W x C).
    occlusion_heatmap (Tensor): A heatmap (H x W) indicating occlusion intensity.

    Returns:
    Tensor: The occluded image.
    """

    # Reshape and replicate the heatmap to match the image dimensions (H x W x C)
    expanded_heatmap = occlusion_heatmap[:, :, None]
    expanded_heatmap = expanded_heatmap.repeat(1, 1, image_tensor.shape[2])

    expanded_heatmap = heatmap_to_binary_mask(expanded_heatmap, torch.mean(expanded_heatmap.float()))

    # Apply the occlusion
    return image_tensor * (expanded_heatmap)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = "WHU-RS19"
    partition = 0.6
    
    test_transform = transforms.Compose(
            [ 
                transforms.ToPILImage(),
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
            ])

    test_dataset = datasets.ImageFolder(root='data/'+dataset+'/test_dataset/')
    partition = 0.6
    
    print(test_dataset.classes)
    
    ''' Load Model '''
    
    model = torch.load('new_saved_models/2023-12-04_vgg16_94.24_proposed_nodiff_WHU-RS19.pth')
    model.to(device)
    model.eval()
    
    # Iterate through DataLoader
    exp_method = Smooth_Score_CAM_GLNet
    
    avg_drop_percentage, conf_increase_percentage = evaluate_exp(test_dataset, test_transform, model, exp_method, device, partition)
    print(f"Average Confidence Drop: {avg_drop_percentage}%")
    print(f"Increase of Confidence: {conf_increase_percentage}%")
    
if __name__ == '__main__':
    main()