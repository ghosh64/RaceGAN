import os
import random
import torchvision.transforms.functional as func
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pickle
from PIL import Image
import cv2
import numpy as np

class dataset(Dataset):
    def __init__(self, train=True, test=False,val=False, train_split=0.8, path='', transform=transforms.ToTensor()):
        self.path=path
        self.trans=transform
        if not (os.path.exists(self.path+'test.txt') or os.path.exists(self.path+'train.txt')):
            self.create_datasets(train_split, path)
        self.train=train
        self.test=test
        self.images=pickle.load(open(path+'test.txt','rb')) if self.test else pickle.load(open(path+'train.txt','rb'))
    
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        data_name = self.images[idx]
        data = process_img(self.path+'Data/' + data_name) # increase contrast of lane
        data = Image.fromarray(data)
        data = data.resize((320, 320))

        label = Image.open(self.path+'Labels/'+data_name) # data name same as label name
        label = func.rgb_to_grayscale(label, num_output_channels=1)
        label = label.resize((320,320))
        
        return self.trans(data), self.trans(label),data_name
    
def process_img(data_name):
    '''
    Increase contrast of input image
    Used as a part of transform in dataloader
    '''
    img = cv2.imread(data_name, cv2.COLOR_BGR2RGB)

    # converting to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    rgb_enhanced_img = np.flip(enhanced_img, 2) # change from BGR to RGB
    
    #save the image here and load directly in the dataloader from pooled/Data/processed/
    return rgb_enhanced_img