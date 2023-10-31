# Convolutional Neural Networks Based Remote Sensing Scene Classification under Clear and Cloudy Environments
Original Paper by Sun et al. accepted by ICCVW 2021

Modified by Ridhwan D.


### Remote Sensing scene image classification under clear and cloudy environments. 
![show example](images/show_example.png)

### Overview architecture of the proposed GLNet for the RS scene classification under clear and cloudy environments.
![archicture](images/architecture.png)



## Required libraries
python 3.6

pytorch 1.0+

numpy

PIl

torchvision


## Usage
1. clone this repo
    ```
    git clone https://github.com/ridhwand/GLNET.git
    ```

2. download the dataset from [google drive](https://drive.google.com/drive/folders/1W2sQAHJ6IKiih8bvrFhMfkujOxNyqjO1?usp=sharing) 

3. train the baseline model
    ```
    python baseline.py
    ```
4. load the model dir you trained in model.py

5. run the training by command 
    ```
    python train.py
    ```

## Citation
    {Huiming Sun, Yuewei Lin, Qin Zou, Shaoyue Song, Jianwu Fang, Hongkai Yu. Convolutional Neural Networks Based Remote Sensing Scene Classification under Clear and Cloudy Environments. IEEE International Conference on Computer Vision Workshop (ICCVW), 2021.}
