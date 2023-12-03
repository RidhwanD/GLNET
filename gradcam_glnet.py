import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from model import SiameseNetwork
from utils import calculate_mid_size
import torch.nn.functional as F
import math

'''
    Reference:
    Livieris, Ioannis E., et al. "Explainable Image Similarity: Integrating Siamese Networks and Grad-CAM." 
    Journal of Imaging 9.10 (2023): 224.
'''

def convert_to_tensor(image:np.ndarray=None, tfms:transforms=None, device:str='cpu')->torch.Tensor:
    '''
        Convert an input image to torch.Tensor

        Parameters
        ----------
        image: input image
        tfms: list of transformations
        device: cpu/cude


        Returns
        -------
        image in torch.Tensor
    '''
    # Apply image transformations
    image = tfms(image).to(device)

    return image


def Grad_CAM_GLNet(image:torch.Tensor, cluster, partition, model:SiameseNetwork=None,\
image_size:tuple=None, figname:str=None, figsize:tuple=(3,3)):
    '''
        Calculates the heatmap for Gram-CAM procedure

        Parameters
        ----------
        image: requested image
        model: Similarity model (Siamese network)
        sub_network: requested sub_network 1 or 2
        image_size: image size
        figname: figure name (optional)
        fisize: figure size (optional)

        Returns
        -------
        heatmap
    '''
    
    # pull the gradients out of the model
    gradients = [model.get_activations_gradient(sub_network=1)]
    for gradient in model.get_activations_gradient(sub_network=2):
        gradients.append(gradient)

    # pool the gradients across the channels
    pooled_gradients = [torch.mean(gradient, dim=[0,2,3]) for gradient in gradients]

    # get the activations of the last convolutional layer
    activations = [model.get_activations_lower(image).detach()]
    for activation in [model.get_activations_upper(cluster[i]).detach() for i in range(len(cluster))]:
        activations.append(activation)

    # weight the channels by corresponding gradients
    for j in range(len(pooled_gradients)):
        for i in range(activations[j].shape[1]):
            activations[j][:, i, :, :] *= pooled_gradients[j][i]
        
    # average the channels of the activations
    heatmaps = []
    for i in range(len(gradients)):
        heatmap = torch.mean(activations[i], dim=1).squeeze()
        heatmap = heatmap.squeeze()
        heatmap = heatmap.detach().cpu().numpy()
        if i == 0:
            size = image_size
        elif i < len(gradients)-1:
            size = (int(0.6*image_size[0]),int(0.6*image_size[1]))
        else:
            size = (calculate_mid_size(image_size, 0.6))
        heatmap = cv2.resize(heatmap, size)
        heatmaps.append(heatmap)
    
    w = image_size[0] - int(partition*image_size[0])
    h = image_size[1] - int(partition*image_size[1])
    m_w, m_h = calculate_mid_size(image_size, partition)
    m_w, m_h = int((image_size[0] - m_w) / 2), int((image_size[1] - m_h) / 2)
    k = 5
    alphaX = 0
    heatmaps[1] = cv2.GaussianBlur(np.pad(heatmaps[1], ((0, h), (0, h)), 'constant', constant_values=0), (k, k), alphaX)
    heatmaps[2] = cv2.GaussianBlur(np.pad(heatmaps[2], ((0, h), (w, 0)), 'constant', constant_values=0), (k, k), alphaX)
    heatmaps[3] = cv2.GaussianBlur(np.pad(heatmaps[3], ((w, 0), (w, 0)), 'constant', constant_values=0), (k, k), alphaX)
    heatmaps[4] = cv2.GaussianBlur(np.pad(heatmaps[4], ((w, 0), (0, h)), 'constant', constant_values=0), (k, k), alphaX)
    heatmaps[5] = cv2.GaussianBlur(np.pad(heatmaps[5], ((m_w, m_h), (m_w, m_h)), 'constant', constant_values=0), (k, k), alphaX)
    
    heatmaps = [torch.from_numpy(heatmap) for heatmap in heatmaps]
    
    heatmaps = torch.stack(heatmaps)
    heatmap = combineHeatmap(heatmaps, "weighted", (0.5, 0.1, 0.1, 0.1, 0.1, 0.1))
    
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = torch.where(heatmap > 0, heatmap, 0)
    #print("Heatmap", heatmap)
    
    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    
    # Reshape & Convert Tensor to numpy
    heatmap = heatmap.squeeze()
    heatmap = heatmap.detach().cpu().numpy()

    # Convert to [0,255]
    heatmap = np.uint8(255 * heatmap)


    if figname is not None:
        plt.figure(figsize=figsize);
        fig = plt.imshow(heatmap);
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig(figname, dpi=300, format='png', 
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None, 
            )

    return heatmap


def Grad_CAMpp_GLNet(image:torch.Tensor, cluster, partition, model:SiameseNetwork=None,\
image_size:tuple=None, figname:str=None, figsize:tuple=(3,3)):
    '''
        Adapted from https://github.com/1Konny/gradcam_plus_plus-pytorch
        Calculates the heatmap for Gram-CAM procedure

        Parameters
        ----------
        image: requested image
        cluster: image cluster
        class_idx: index of resulting class
        model: Similarity model (Siamese network)
        sub_network: requested sub_network 1 or 2
        image_size: image size
        figname: figure name (optional)
        fisize: figure size (optional)

        Returns
        -------
        heatmap
    '''
    b, c, h, w = image.size()
    with torch.no_grad():
        logit = model(image, cluster)
    score = logit[:, logit.max(1)[-1]].squeeze()
    
    # pull the gradients out of the model
    gradients = [model.get_activations_gradient(sub_network=1)]
    for gradient in model.get_activations_gradient(sub_network=2):
        gradients.append(gradient)

    # get the activations of the last convolutional layer
    activations = [model.get_activations_lower(image).detach()]
    for activation in [model.get_activations_upper(cluster[i]).detach() for i in range(len(cluster))]:
        activations.append(activation)
    
    b, k, u, v = gradients[0].size()

    for j in range(len(gradients)):
        alpha_num = gradients[j].pow(2)
        alpha_denom = gradients[j].pow(2).mul(2) + \
                activations[j].mul(gradients[j].pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients_upper = F.relu(score.exp()*gradients[j]) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights_upper = (alpha*positive_gradients_upper).view(b, k, u*v).sum(-1) # .view(b, k, 1, 1)

        # weight the channels by corresponding gradients
        for i in range(activations[j].shape[1]):
            activations[j][:, i, :, :] *= weights_upper[0][i]    
            
    # average the channels of the activations
    heatmaps = []
    for i in range(len(gradients)):
        heatmap = torch.mean(activations[i], dim=1).squeeze()
        heatmap = heatmap.squeeze()
        heatmap = heatmap.detach().cpu().numpy()
        if i == 0:
            size = image_size
        elif i < len(gradients)-1:
            size = (int(0.6*image_size[0]),int(0.6*image_size[1]))
        else:
            size = (calculate_mid_size(image_size, 0.6))
        heatmap = cv2.resize(heatmap, size)
        heatmaps.append(heatmap)
    
    w = image_size[0] - int(partition*image_size[0])
    h = image_size[1] - int(partition*image_size[1])
    m_w, m_h = calculate_mid_size(image_size, partition)
    m_w, m_h = int((image_size[0] - m_w) / 2), int((image_size[1] - m_h) / 2)
    k = 5
    alphaX = 0
    heatmaps[1] = cv2.GaussianBlur(np.pad(heatmaps[1], ((0, h), (0, h)), 'constant', constant_values=0), (k, k), alphaX)
    heatmaps[2] = cv2.GaussianBlur(np.pad(heatmaps[2], ((0, h), (w, 0)), 'constant', constant_values=0), (k, k), alphaX)
    heatmaps[3] = cv2.GaussianBlur(np.pad(heatmaps[3], ((w, 0), (w, 0)), 'constant', constant_values=0), (k, k), alphaX)
    heatmaps[4] = cv2.GaussianBlur(np.pad(heatmaps[4], ((w, 0), (0, h)), 'constant', constant_values=0), (k, k), alphaX)
    heatmaps[5] = cv2.GaussianBlur(np.pad(heatmaps[5], ((m_w, m_h), (m_w, m_h)), 'constant', constant_values=0), (k, k), alphaX)
    
    heatmaps = [torch.from_numpy(heatmap) for heatmap in heatmaps]
    
    heatmaps = torch.stack(heatmaps)
    heatmap = combineHeatmap(heatmaps, "weighted", (0.5, 0.1, 0.1, 0.1, 0.1, 0.1))
    
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = torch.where(heatmap > 0, heatmap, 0)
    
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # Reshape & Convert Tensor to numpy
    heatmap = heatmap.squeeze()
    heatmap = heatmap.detach().cpu().numpy()


    # Resize Heatmap
    heatmap = cv2.resize(heatmap, image_size)
    # Convert to [0,255]
    heatmap = np.uint8(255 * heatmap)


    if figname is not None:
        plt.figure(figsize=figsize);
        fig = plt.imshow(heatmap);
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig(figname, dpi=300, format='png', 
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None, 
            )

    return heatmap

def Score_CAM_GLNet(image: torch.Tensor, cluster, model: SiameseNetwork, target_class, partition, device, image_size=None, figname=None, figsize=(3,3)):
    '''
    Generate a Score-CAM heatmap

    Parameters
    ----------
    image: Input image tensor
    model: The neural network model
    cluster: Image cluster
    target_class: Target class index
    image_size: Size to resize the heatmap (optional)
    figname: Filename to save the figure (optional)
    figsize: Size of the figure (optional)

    Returns
    -------
    heatmap
    '''
    # get the activations of the last convolutional layer
    activations = [model.get_activations_lower(image).detach()]
    for activation in [model.get_activations_upper(cluster[i]).detach() for i in range(len(cluster))]:
        activations.append(activation)

    # Initialize scores
    scores = torch.zeros(activations[0].shape[1], dtype=torch.float32)

    # Get the size of the original image
    original_size = image.shape[-2:]
    
    N = 10  # Number of top activation maps to consider
    for activation in activations:
        activation = activation[:, activation.mean(dim=(2, 3)).topk(N, dim=1).indices.squeeze(), :, :]
    
    # Iterate over each activation map
    for i in range(activations[0].shape[1]):
        # print("Process ",i+1, " of ", activations.shape[1])
        # Upsample activation to match the size of the original image
        upsampled_activation = []
        for activation in activations:
            upsampled_activation.append(F.interpolate(activation[:, i:i+1, :, :], size=original_size, mode='bilinear', align_corners=False))

        # Mask the input image with upsampled activation map
        masked_input = image * upsampled_activation[0]
        masked_cluster = []
        for idx, img in enumerate(cluster):
            masked_cluster.append(img * upsampled_activation[idx])

        # Forward pass with masked input
        with torch.no_grad():
            output = model(masked_input, masked_cluster)
        torch.cuda.empty_cache()

        # Record the score for the target class
        scores[i] = output[0, target_class]

    # Normalize scores
    scores = scores - scores.min()
    scores = scores / scores.max()

    scores = scores.to(device)

    # Weight and combine activation maps
    weighted_maps = [activation * scores.view(-1, 1, 1, 1) for activation in activations]
    heatmaps = [weighted_map.sum(0) for weighted_map in weighted_maps]

    # Apply ReLU and normalize heatmap
    # heatmaps = [F.relu(heatmap) for heatmap in heatmaps]
    # for heatmap in heatmaps:
    #     heatmap /= heatmap.max()
    

    # Post-processing steps similar to Grad-CAM.
    
    # Reshape & Convert Tensor to numpy
    heatmaps = [torch.mean(heatmap, 0) for heatmap in heatmaps]
    
    new_heatmaps = []
    for i in range(len(heatmaps)):
        heatmap = heatmaps[i].squeeze()
        heatmap = heatmap.detach().cpu().numpy()
        if i == 0:
            size = image_size
        elif i < len(heatmaps)-1:
            size = (int(0.6*image_size[0]),int(0.6*image_size[1]))
        else:
            size = (calculate_mid_size(image_size, 0.6))
        heatmap = cv2.resize(heatmap, size)
        new_heatmaps.append(heatmap)
    
    w = image_size[0] - int(partition*image_size[0])
    h = image_size[1] - int(partition*image_size[1])
    m_w, m_h = calculate_mid_size(image_size, partition)
    m_w, m_h = int((image_size[0] - m_w) / 2), int((image_size[1] - m_h) / 2)
    k = 5
    alphaX = 0
    new_heatmaps[1] = cv2.GaussianBlur(np.pad(new_heatmaps[1], ((0, h), (0, h)), 'constant', constant_values=0), (k, k), alphaX)
    new_heatmaps[2] = cv2.GaussianBlur(np.pad(new_heatmaps[2], ((0, h), (w, 0)), 'constant', constant_values=0), (k, k), alphaX)
    new_heatmaps[3] = cv2.GaussianBlur(np.pad(new_heatmaps[3], ((w, 0), (w, 0)), 'constant', constant_values=0), (k, k), alphaX)
    new_heatmaps[4] = cv2.GaussianBlur(np.pad(new_heatmaps[4], ((w, 0), (0, h)), 'constant', constant_values=0), (k, k), alphaX)
    new_heatmaps[5] = cv2.GaussianBlur(np.pad(new_heatmaps[5], ((m_w, m_h), (m_w, m_h)), 'constant', constant_values=0), (k, k), alphaX)
    
    heatmaps = [torch.from_numpy(heatmap) for heatmap in new_heatmaps]
    
    heatmaps = torch.stack(heatmaps)
    heatmap = combineHeatmap(heatmaps, "weighted", (0.5, 0.1, 0.1, 0.1, 0.1, 0.1))
    
    heatmap = F.relu(heatmap)
    heatmap /= heatmap.max()
    
    # heatmap = heatmap.squeeze()
    heatmap = heatmap.detach().cpu().numpy()

    # Convert to [0,255]
    heatmap = np.uint8(255 * heatmap)
    
    
    if figname is not None:
        plt.figure(figsize=figsize);
        fig = plt.imshow(heatmap);
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig(figname, dpi=300, format='png', 
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None, 
            )
    
    return heatmap

def Smooth_Score_CAM_GLNet(image: torch.Tensor, cluster, model, target_class, partition, device, image_size=None, figname:str=None, figsize=(3,3), num_samples=20, std_dev=0.15):
    '''
    Generate a Smooth Score-CAM heatmap

    Parameters
    ----------
    image: Input image tensor
    model: The neural network model
    target_layer: The target convolutional layer
    target_class: Target class index
    num_samples: Number of samples with noise
    std_dev: Standard deviation for noise

    Returns
    -------
    Averaged heatmap
    '''
    heatmaps = []

    for i in range(num_samples):
        # Add random noise to the image
        print(i)
        noisy_image = image + torch.randn(image.shape).to(device) * std_dev
        noisy_cluster = []
        for img in cluster:
            noisy_cluster.append(img + torch.randn(img.shape).to(device) * std_dev)

        # Generate heatmap for noisy image
        heatmap = Score_CAM_GLNet(noisy_image, noisy_cluster, model, target_class, partition, device, image_size)
        heatmaps.append(heatmap)

    mean_heatmap = combineHeatmapNP(np.array(heatmaps), 'mean')
    
    if figname is not None:
        plt.figure(figsize=figsize);
        fig = plt.imshow(mean_heatmap);
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig(figname, dpi=300, format='png', 
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None, 
            )

    return mean_heatmap

def Smooth_Grad_CAMpp_GLNet(image: torch.Tensor, cluster, model, partition, device, image_size=None, figname:str=None, figsize=(3,3), num_samples=20, std_dev=0.15):
    '''
    Generate a Smooth Score-CAM heatmap

    Parameters
    ----------
    image: Input image tensor
    model: The neural network model
    target_layer: The target convolutional layer
    target_class: Target class index
    num_samples: Number of samples with noise
    std_dev: Standard deviation for noise

    Returns
    -------
    Averaged heatmap
    '''
    heatmaps = []
    
    for i in range(num_samples):
        # Add random noise to the image
        noisy_image = image + torch.randn(image.shape).to(device) * std_dev
        noisy_cluster = []

        for img in cluster:
            noisy_cluster.append(img + torch.randn(img.shape).to(device) * std_dev)
        # Generate heatmap for noisy image
        heatmap = Grad_CAMpp_GLNet(noisy_image, noisy_cluster, partition, model, image_size)
        heatmaps.append(heatmap)

    # Average heatmaps
    mean_heatmap = combineHeatmapNP(np.array(heatmaps), 'mean')
    
    if figname is not None:
        plt.figure(figsize=figsize);
        fig = plt.imshow(mean_heatmap);
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig(figname, dpi=300, format='png', 
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None, 
            )

    return mean_heatmap

def combineHeatmapNP(heatmaps, mode='max'):
    '''
        Combine several heatmaps into one

        Parameters
        ----------
        heatmaps: an np.array containing heatmaps to combine
        mode: mode of combination

        Returns
        -------
        combined heatmap
    '''
    if (mode == 'max'):
        final_heatmap = np.amax(heatmaps, axis=0)
    elif (mode == 'mean'):
        final_heatmap = np.mean(heatmaps, axis=0)
    final_heatmap = np.uint8(final_heatmap.astype(int))
    
    
    fig = plt.imshow(final_heatmap);
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    
    return final_heatmap

def combineHeatmap(heatmaps, mode='max', weights=None):
    '''
        Combine several heatmaps into one

        Parameters
        ----------
        heatmaps: a Tensor containing heatmaps to combine
        mode: mode of combination
        weight: list of weight. Must sum to 1, must be of same size with heatmaps

        Returns
        -------
        combined heatmap
    '''
    if (mode == 'max'):
        final_heatmap, _ = torch.max(heatmaps, dim=0)
    elif (mode == 'mean'):
        final_heatmap = torch.mean(heatmaps, dim=0)
    elif (mode == 'meannz'):
        final_heatmap = meanNZ(heatmaps)
    elif (mode == 'weighted'):
        if weights == None or (not math.isclose(sum(weights), 1, rel_tol=1e-9)) or len(weights) != len(heatmaps):
            raise ValueError("Invalid Weight")
        else:
            weights_tensor = torch.tensor(weights).view(len(weights), 1, 1)
            final_heatmap = torch.sum(heatmaps * weights_tensor, dim=0)
    
    return final_heatmap

def meanNZ(heatmaps):
    non_zero_mask = heatmaps != 0
    sum_non_zero = torch.sum(heatmaps * non_zero_mask, dim=0)
    count_non_zero = torch.sum(non_zero_mask, dim=0)
    mean_tensor = sum_non_zero / count_non_zero.where(count_non_zero != 0, torch.tensor(1.0))
    return mean_tensor


def Median_Grad_CAM_GLNet(image:torch.Tensor, cluster, partition, model:SiameseNetwork=None,\
image_size:tuple=None, figname:str=None, figsize:tuple=(3,3)):
    '''
        Adapted from https://github.com/1Konny/gradcam_plus_plus-pytorch
        Calculates the heatmap for Gram-CAM procedure

        Parameters
        ----------
        image: requested image
        cluster: image cluster
        class_idx: index of resulting class
        model: Similarity model (Siamese network)
        sub_network: requested sub_network 1 or 2
        image_size: image size
        figname: figure name (optional)
        fisize: figure size (optional)

        Returns
        -------
        heatmap
    '''
    b, c, h, w = image.size()
    with torch.no_grad():
        logit = model(image, cluster)
    score = logit[:, logit.max(1)[-1]].squeeze()
    
    # pull the gradients out of the model
    gradients = [model.get_activations_gradient(sub_network=1)]
    for gradient in model.get_activations_gradient(sub_network=2):
        gradients.append(gradient)

    # get the activations of the last convolutional layer
    activations = [model.get_activations_lower(image).detach()]
    for activation in [model.get_activations_upper(cluster[i]).detach() for i in range(len(cluster))]:
        activations.append(activation)
    
    b, k, u, v = gradients[0].size()

    for j in range(len(gradients)):
        positive_gradients = F.relu(score.exp()*gradients[j]) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = torch.median(positive_gradients.view(b, k, u*v), dim=-1).values
    
        # weight the channels by corresponding gradients
        for i in range(activations[j].shape[1]):
            activations[j][:, i, :, :] *= weights[0][i]    
        #print("Activations", activations)
        
    # average the channels of the activations
    heatmaps = []
    for i in range(len(gradients)):
        heatmap = torch.mean(activations[i], dim=1).squeeze()
        heatmap = heatmap.squeeze()
        heatmap = heatmap.detach().cpu().numpy()
        if i == 0:
            size = image_size
        elif i < len(gradients)-1:
            size = (int(0.6*image_size[0]),int(0.6*image_size[1]))
        else:
            size = (calculate_mid_size(image_size, 0.6))
        heatmap = cv2.resize(heatmap, size)
        heatmaps.append(heatmap)
    
    w = image_size[0] - int(partition*image_size[0])
    h = image_size[1] - int(partition*image_size[1])
    m_w, m_h = calculate_mid_size(image_size, partition)
    m_w, m_h = int((image_size[0] - m_w) / 2), int((image_size[1] - m_h) / 2)
    k = 5
    alphaX = 0
    heatmaps[1] = cv2.GaussianBlur(np.pad(heatmaps[1], ((0, h), (0, h)), 'constant', constant_values=0), (k, k), alphaX)
    heatmaps[2] = cv2.GaussianBlur(np.pad(heatmaps[2], ((0, h), (w, 0)), 'constant', constant_values=0), (k, k), alphaX)
    heatmaps[3] = cv2.GaussianBlur(np.pad(heatmaps[3], ((w, 0), (w, 0)), 'constant', constant_values=0), (k, k), alphaX)
    heatmaps[4] = cv2.GaussianBlur(np.pad(heatmaps[4], ((w, 0), (0, h)), 'constant', constant_values=0), (k, k), alphaX)
    heatmaps[5] = cv2.GaussianBlur(np.pad(heatmaps[5], ((m_w, m_h), (m_w, m_h)), 'constant', constant_values=0), (k, k), alphaX)
    
    heatmaps = [torch.from_numpy(heatmap) for heatmap in heatmaps]
    
    heatmaps = torch.stack(heatmaps)
    heatmap = combineHeatmap(heatmaps, "weighted", (0.5, 0.1, 0.1, 0.1, 0.1, 0.1))
    
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = torch.where(heatmap > 0, heatmap, 0)
    #print("Heatmap", heatmap)
    
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # Reshape & Convert Tensor to numpy
    heatmap = heatmap.squeeze()
    heatmap = heatmap.detach().cpu().numpy()

    # Convert to [0,255]
    heatmap = np.uint8(255 * heatmap)


    if figname is not None:
        plt.figure(figsize=figsize);
        fig = plt.imshow(heatmap);
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig(figname, dpi=300, format='png', 
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None, 
            )

    return heatmap




