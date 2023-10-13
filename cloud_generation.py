import os
import random
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import numpy as np
import skimage.transform
import cv2

def generate_cloud(im_size_h, im_size_w,k = 2,):
    # Initialize the white noise pattern
    base_pattern = np.random.uniform(0,255, (im_size_h//2, im_size_w//2))

    # Initialize the output pattern
    turbulence_pattern = np.zeros((im_size_h, im_size_w))

    # Create cloud pattern
    power_range = [k**i for i in range(2, int(np.log2(min(im_size_h, im_size_w))))]
    
    for p in power_range:
        quadrant = base_pattern[:p, :p]
        upsampled_pattern = skimage.transform.resize(quadrant, (im_size_h, im_size_w), mode='reflect')
        turbulence_pattern += upsampled_pattern / float(p)

    turbulence_pattern /= sum([1 / float(p) for p in power_range])    
    return turbulence_pattern


def add_cloud(file_name, file_name_save, k):
    img = cv2.imread(file_name)
    im_size_h, im_size_w = np.shape(img)[:2]
        
    # Generate cloud map
    cloud_map = generate_cloud(im_size_h, im_size_w,k)
    fourground_map = (255 - cloud_map) / 255
    res = np.zeros((np.shape(img)))
    res[:,:,0] = (img[:,:,0] * fourground_map + cloud_map) / 256
    res[:,:,1] = (img[:,:,1] * fourground_map + cloud_map) / 256
    res[:,:,2] = (img[:,:,2] * fourground_map + cloud_map) / 256
    
    plt.imsave(file_name_save.replace('.jpg', '_cloud.png'), res)
    print(file_name_save.replace('.jpg', '_cloud.png'))
    
    return cloud_map, res.astype(np.uint8),fourground_map

def main():
    data = "NWPU-RESISC45"
    dataPath_test = "data/"+data+"/test_dataset-ori"
    dataPath_test_cloudy = "data/"+data+"/test_dataset"
    dataPath_train = "data/"+data+"/train_dataset-ori"
    dataPath_train_cloudy = "data/"+data+"/train_dataset"
    label = os.listdir(dataPath_test)

    for l in label:
        imgs = os.listdir(dataPath_test+"/"+l)
        for img in imgs:
            add_cloud(dataPath_test+"/"+l+"/"+img,dataPath_test_cloudy+"/"+l+"/"+img,2)
        imgs = os.listdir(dataPath_train+"/"+l)
        for img in imgs:
            add_cloud(dataPath_train+"/"+l+"/"+img,dataPath_train_cloudy+"/"+l+"/"+img,2)

if __name__ == '__main__':
    main()