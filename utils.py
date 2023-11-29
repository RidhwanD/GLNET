# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def imshow(image:torch.Tensor=None, heatmap:np.ndarray=None, scale:float=0.4, mean:tuple=(0, 0, 0), std:tuple=(1, 1, 1), figsize:tuple=(10,10), figname:str=None):
    '''
        Reference:
        Livieris, Ioannis E., et al. "Explainable Image Similarity: Integrating Siamese Networks and Grad-CAM." 
        Journal of Imaging 9.10 (2023): 224.
        
        
        Show and save image
        
        Parameters
        ----------
        image: image as torch.Tensor
        heatmap: Grad-CAM heatmap (optional)
        scale: Scaling parameters for merging Grad-CAM heatmap with image
    '''
    # Convert image to NumPy
    npimg = image.squeeze(0).cpu().numpy()

    # Image normalization
    npimg = npimg * np.array(std)[:,None,None] + np.array(mean)[:,None,None]  # Unnormalize

    # Reshape
    npimg = np.transpose(npimg, (1, 2, 0))

    if heatmap is not None:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) / 255.0
        npimg = heatmap * scale + npimg
        

    # Show image
    plt.figure(figsize=figsize)
    fig = plt.imshow(npimg)
    if figname is not None:
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig(figname, dpi=300, 
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None, 
            )        

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def generateCluster(img, transform, partion=0.6, size=256):
    width, length = int(size*partion),int(size*partion)
    upper_left   =   cv2.resize(img[:width,:length],(size,size))
    upper_right  =   cv2.resize(img[:width,-length:],(size,size))
    bottom_right =   cv2.resize(img[-width:,-length:],(size,size))
    bottom_left  =   cv2.resize(img[-width:,:length],(size,size))
    mid = img[int((1-partion)/2*(img.shape[0])):-int((1-partion)/2*(img.shape[0])),int((1-partion)/2*(img.shape[1])):-int((1-partion)/2*(img.shape[1]))]
    cluster_data = upper_left,upper_right,bottom_right,bottom_left,mid
    img = transform(img)
    cluster_data = [transform(i) for i in cluster_data]
    # label = self.imgs[index][1]
    
    return img, cluster_data