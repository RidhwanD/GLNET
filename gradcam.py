import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from model import SiameseNetwork
from utils import calculate_mid_size
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


    
def Grad_CAM_heatmap(image:torch.Tensor, model:SiameseNetwork=None, sub_network:int=None,\
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
    gradients = model.get_activations_gradient(sub_network=sub_network)
    #print(gradients)

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0,2,3])
    #print("Pooled Grads", pooled_gradients)

    # get the activations of the last convolutional layer
    activations = model.get_activations(image).detach()

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


def Grad_CAM_plusplus(image:torch.Tensor, cluster, class_idx, model:SiameseNetwork=None, sub_network:int=None,\
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

    logit = model(image, cluster)
    if class_idx is None:
        score = logit[:, logit.max(1)[-1]].squeeze()
    else:
        score = logit[:, class_idx].squeeze() 
    
    # pull the gradients out of the model
    gradients = model.get_activations_gradient(sub_network=sub_network)
    #print(gradients)

    # get the activations of the last convolutional layer
    activations = model.get_activations(image).detach()
    
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
    target_class: Target class index
    image_size: Size to resize the heatmap (optional)
    figname: Filename to save the figure (optional)
    figsize: Size of the figure (optional)

    Returns
    -------
    heatmap
    '''
    # get the activations of the last convolutional layer
    activations = model.get_activations(image).detach()

    # Initialize scores
    scores = torch.zeros(activations.shape[1], dtype=torch.float32)

    # Get the size of the original image
    original_size = image.shape[-2:]
    
    N = 10  # Number of top activation maps to consider
    activations = activations[:, activations.mean(dim=(2, 3)).topk(N, dim=1).indices.squeeze(), :, :]
    
    # Iterate over each activation map
    for i in range(activations.shape[1]):
        print("Process ",i+1, " of ", activations.shape[1])
        # Upsample activation to match the size of the original image
        upsampled_activation = F.interpolate(activations[:, i:i+1, :, :], size=original_size, mode='bilinear', align_corners=False)

        # Mask the input image with upsampled activation map
        masked_input = image * upsampled_activation

        # Forward pass with masked input
        output = model(masked_input, cluster)

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

def Grad_CAM_GLNet(img, cluster, size, partition, model):
    heatmap = Grad_CAM_heatmap(img, model, 1, (size[0],size[1]),'F')
    all_heatmaps = [heatmap]
    w = size[0] - int(partition*size[0])
    h = size[1] - int(partition*size[1])
    m_w, m_h = calculate_mid_size(size, partition)
    m_w, m_h = int((size[0] - m_w) / 2), int((size[1] - m_h) / 2)
    all_heatmaps.append(np.pad(Grad_CAM_heatmap(cluster[0], model, 1, (int(0.6*size[0]),int(0.6*size[1])),'F'), ((0, h), (0, h)), 'constant', constant_values=0))
    all_heatmaps.append(np.pad(Grad_CAM_heatmap(cluster[1], model, 1, (int(0.6*size[0]),int(0.6*size[1])),'F'), ((0, h), (w, 0)), 'constant', constant_values=0))
    all_heatmaps.append(np.pad(Grad_CAM_heatmap(cluster[2], model, 1, (int(0.6*size[0]),int(0.6*size[1])),'F'), ((w, 0), (w, 0)), 'constant', constant_values=0))
    all_heatmaps.append(np.pad(Grad_CAM_heatmap(cluster[3], model, 1, (int(0.6*size[0]),int(0.6*size[1])),'F'), ((w, 0), (0, h)), 'constant', constant_values=0))
    all_heatmaps.append(np.pad(Grad_CAM_heatmap(cluster[4], model, 1, (calculate_mid_size(size, 0.6)),'F'), ((m_w, m_h), (m_w, m_h)), 'constant', constant_values=0))
    all_heatmaps = np.array(all_heatmaps)
    final_heatmap = combineHeatmap(all_heatmaps,'max')
    
    return final_heatmap