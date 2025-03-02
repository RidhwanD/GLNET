import os
import random
import shutil
import matplotlib.pyplot as plt
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
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_size_h, im_size_w = np.shape(img_rgb)[:2]
        
    # Generate cloud map
    cloud_map = generate_cloud(im_size_h, im_size_w,k)
    fourground_map = (255 - cloud_map) / 255
    res = np.zeros((np.shape(img_rgb)))
    res[:,:,0] = (img_rgb[:,:,0] * fourground_map + cloud_map) / 255
    res[:,:,1] = (img_rgb[:,:,1] * fourground_map + cloud_map) / 255
    res[:,:,2] = (img_rgb[:,:,2] * fourground_map + cloud_map) / 255
    
    window_name = 'image'
    cv2.imshow(window_name, img_rgb)
    
    
    plt.imsave(file_name_save.replace('.jpg', '_cloud.png'), res)
    
    return cloud_map, res.astype(np.uint8),fourground_map

def main():
    mode = "mixed" # 'all': all cloudy, 'mixed': mixed of cloudy and clear with c_perc% cloudy
    c_perc = 50  # percentage of cloudy image for mixed mode only
    dataset = "rsscn7"
    
    print("Generating",mode,"cloudy",dataset)

    dataPath_test = os.path.join("data", dataset, "test_dataset-clear")
    dataPath_test_cloudy = os.path.join("data", dataset, "test_dataset")
    dataPath_train = os.path.join("data", dataset, "train_dataset-clear")
    dataPath_train_cloudy = os.path.join("data", dataset, "train_dataset")
    
    label = os.listdir(dataPath_test)

    for l in label:
        if (mode == "all"):
            imgs = os.listdir(os.path.join(dataPath_test, l))
            if not os.path.exists(os.path.join(dataPath_test_cloudy, l)):
                os.mkdir(os.path.join(dataPath_test_cloudy, l))
            for img in imgs:
                add_cloud(os.path.join(dataPath_test, l, img),os.path.join(dataPath_test_cloudy, l, img),2)
            
            imgs = os.listdir(os.path.join(dataPath_train, l))
            if not os.path.exists(os.path.join(dataPath_train_cloudy, l)):
                os.mkdir(os.path.join(dataPath_train_cloudy, l))
            for img in imgs:
                add_cloud(os.path.join(dataPath_train, l, img),os.path.join(dataPath_train_cloudy, l, img),2)
        elif (mode == "mixed"):
            imgs = os.listdir(os.path.join(dataPath_test, l))
            num_test = len([name for name in imgs if os.path.isfile(os.path.join(dataPath_test, l, name))])
            test_num = int(c_perc/100 * num_test)
            idx_test = random.sample(range(1, num_test), test_num)
            if not os.path.exists(os.path.join(dataPath_test_cloudy, l)):
                os.mkdir(os.path.join(dataPath_test_cloudy, l))
            idx = 1
            for img in imgs:
                if idx in idx_test:
                    add_cloud(os.path.join(dataPath_test, l, img),os.path.join(dataPath_test_cloudy, l, img),2)
                else:
                    shutil.copyfile(os.path.join(dataPath_test, l, img),
                        os.path.join(dataPath_test_cloudy, l, img))
                idx += 1
            
            imgs = os.listdir(os.path.join(dataPath_train, l))
            num_train = len([name for name in imgs if os.path.isfile(os.path.join(dataPath_train, l, name))])
            train_num = int(c_perc/100 * num_train)
            idx_train = random.sample(range(1, num_train), train_num)
            if not os.path.exists(os.path.join(dataPath_train_cloudy, l)):
                os.mkdir(os.path.join(dataPath_train_cloudy, l))
            idx = 1
            for img in imgs:
                if idx in idx_train:
                    add_cloud(os.path.join(dataPath_train, l, img),os.path.join(dataPath_train_cloudy, l, img),2)
                else:
                    shutil.copyfile(os.path.join(dataPath_train, l, img),
                        os.path.join(dataPath_train_cloudy, l, img))
                idx += 1

if __name__ == '__main__':
    main()