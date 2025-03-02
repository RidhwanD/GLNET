import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from model import SiameseNetwork
import torch.nn.functional as F

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


    
def Grad_CAM(image:torch.Tensor, model:SiameseNetwork=None,\
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
    gradients = model.get_activations_gradient(sub_network=1)
    #print(gradients)

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0,2,3])
    #print("Pooled Grads", pooled_gradients)

    # get the activations of the last convolutional layer
    activations = model.get_activations_lower(image).detach()

    # weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]    
    #print("Activations", activations)
        
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    #print("Heatmap", heatmap)
    
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = torch.where(heatmap > 0, heatmap, 0)
    #print("Heatmap", heatmap)
    
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # Reshape & Convert Tensor to numpy
    heatmap = heatmap.squeeze()
    heatmap = heatmap.detach().cpu().numpy()

    print(heatmap.shape)

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


def Grad_CAMpp(image:torch.Tensor, cluster, model:SiameseNetwork=None,\
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
    gradients = model.get_activations_gradient(sub_network=1)
    #print(gradients)

    # get the activations of the last convolutional layer
    activations = model.get_activations_lower(image).detach()
    
    b, k, u, v = gradients.size()

    alpha_num = gradients.pow(2)
    alpha_denom = gradients.pow(2).mul(2) + \
            activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

    alpha = alpha_num.div(alpha_denom+1e-7)
    positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
    weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1) # .view(b, k, 1, 1)

    # weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= weights[0][i]    
    #print("Activations", activations)
        
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    #print("Heatmap", heatmap)
    
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = torch.where(heatmap > 0, heatmap, 0)
    #print("Heatmap", heatmap)
    
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

def Score_CAM(image: torch.Tensor, cluster, model: SiameseNetwork, target_class, device, image_size=None, figname=None, figsize=(3,3)):
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
    activations = model.get_activations_lower(image).detach()

    # Initialize scores
    scores = torch.zeros(activations.shape[1], dtype=torch.float32)

    # Get the size of the original image
    original_size = image.shape[-2:]
    
    N = 10  # Number of top activation maps to consider
    activations = activations[:, activations.mean(dim=(2, 3)).topk(N, dim=1).indices.squeeze(), :, :]
    
    # Iterate over each activation map
    for i in range(activations.shape[1]):
        # print("Process ",i+1, " of ", activations.shape[1])
        # Upsample activation to match the size of the original image
        upsampled_activation = F.interpolate(activations[:, i:i+1, :, :], size=original_size, mode='bilinear', align_corners=False)

        # Mask the input image with upsampled activation map
        masked_input = image * upsampled_activation

        # Forward pass with masked input
        with torch.no_grad():
            output = model(masked_input, cluster)
        torch.cuda.empty_cache()

        # Record the score for the target class
        scores[i] = output[0, target_class]

    # Normalize scores
    scores = scores - scores.min()
    scores = scores / scores.max()

    scores = scores.to(device)

    # Weight and combine activation maps
    weighted_maps = activations * scores.view(-1, 1, 1, 1)
    heatmap = weighted_maps.sum(0)

    # Apply ReLU and normalize heatmap
    heatmap = F.relu(heatmap)
    heatmap /= heatmap.max()

    # Post-processing steps similar to Grad-CAM.
    
    # Reshape & Convert Tensor to numpy
    heatmap = torch.mean(heatmap, 0)
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

def Smooth_Score_CAM(image: torch.Tensor, cluster, model, target_class, device, image_size=None, figname:str=None, figsize=(3,3), num_samples=50, std_dev=0.15):
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

    for _ in range(num_samples):
        # Add random noise to the image
        noisy_image = image + torch.randn(image.shape).to(device) * std_dev
        noisy_cluster = []
        for img in cluster:
            noisy_cluster.append(img + torch.randn(img.shape).to(device) * std_dev)

        # Generate heatmap for noisy image
        heatmap = Score_CAM(noisy_image, noisy_cluster, model, target_class, device, image_size)
        heatmaps.append(heatmap)

    mean_heatmap = combineHeatmap(np.array(heatmaps), 'mean')
    
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

def Smooth_Grad_CAMpp(image: torch.Tensor, cluster, model, device, image_size=None, figname:str=None, figsize=(3,3), num_samples=50, std_dev=0.15):
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
        heatmap = Grad_CAMpp(noisy_image, noisy_cluster, model, image_size)
        heatmaps.append(heatmap)

    # Average heatmaps
    mean_heatmap = combineHeatmap(np.array(heatmaps), 'mean')
    
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

def combineHeatmap(heatmaps, mode='max'):
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

def Median_Grad_CAM(image:torch.Tensor, cluster, model:SiameseNetwork=None,\
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
    gradients = model.get_activations_gradient(sub_network=1)
    #print(gradients)

    # get the activations of the last convolutional layer
    activations = model.get_activations_lower(image).detach()
    
    b, k, u, v = gradients.size()

    positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
    weights = torch.median(positive_gradients.view(b, k, u*v), dim=-1).values

    # weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= weights[0][i]    
    #print("Activations", activations)
        
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    #print("Heatmap", heatmap)
    
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = torch.where(heatmap > 0, heatmap, 0)
    #print("Heatmap", heatmap)
    
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # Reshape & Convert Tensor to numpy
    heatmap = heatmap.squeeze()
    print(heatmap.shape)
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





