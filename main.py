import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from generator import G_model
from discriminator import D_model
from data import dataset
from train_dis import train_D
from train_gen import train_G
from PIL import Image
import sys
import os
import gc 
from tqdm import tqdm

def train(G,
          D,
          batch_size,
          num_epochs,
          device,
          cycles,
          G_weight_path,
          D_weight_path,
          result_path):
    
    optimizer_G = torch.optim.Adam(G.parameters(), weight_decay=1e-2)
    optimizer_D = torch.optim.Adam(D.parameters(), weight_decay=2e-2) # faster decay rate
    
    loss_G = nn.MSELoss() 
    loss_D = nn.MSELoss()
    
    for i in range(cycles):
        # Training dataset
        train_data_obj = dataset()   
        train_data_load = DataLoader(train_data_obj, batch_size=batch_size, shuffle=True, drop_last=True) 
        train_G(G, 
                D, 
                train_data_load,
                loss_G,
                loss_D,
                optimizer_G,
                num_epochs,
                device,
                G_weight_path,
                i)
        
        train_D(G,
                D, 
                train_data_load,
                loss_D,
                optimizer_D,
                num_epochs,
                device,
                D_weight_path,
                i)
        
        if i%2==0 or i==(cycles-1):
            if not os.path.exists(result_path+f'cycle_{i}/'):
                os.makedirs(result_path+f'cycle_{i}/')

def rgb_label(label):
	pred_rgb = np.ones((label.shape[0], label.shape[1], 3))
	for h in range(label.shape[0]):
		for w in range(label.shape[1]):
			if label[h][w][0] != 1: # if the prediction is not white
				pred_rgb[h][w][0] = 1
				pred_rgb[h][w][1] = 0
				pred_rgb[h][w][2] = 0
	return pred_rgb


def black_rgb_label(label):
	pred_rgb = np.ones((label.shape[0], label.shape[1], 3))
	for h in range(label.shape[0]):
		for w in range(label.shape[1]):
			if label[h][w][0] != 1: # if the prediction is not white
				pred_rgb[h][w][0] = 0
				pred_rgb[h][w][1] = 0
				pred_rgb[h][w][2] = 0
	return pred_rgb


def black_rgb_label_init_guess(label):
	pred_rgb = np.ones((label.shape[0], label.shape[1], 3))
	for h in range(label.shape[0]):
		for w in range(label.shape[1]):
			pred_rgb[h][w][0] = label[h][w][0]
			pred_rgb[h][w][1] = label[h][w][0]
			pred_rgb[h][w][2] = label[h][w][0]
	return pred_rgb

'''
insert weight and result paths here
'''
if __name__ == "__main__":
    batch_size = 8 
    num_epochs = 5
    cycles = 20
    is_red_pred = True
    cuda = "cuda:0"
    
    #weight paths
    G_weight_path=''
    D_weight_path=''
    result_path=''
    
    device = torch.device(cuda)
    
    G = G_model(device=device)
    D = D_model(device=device)   
    
    # Normal Training
    train(G,
          D,
          batch_size,
          num_epochs,
          device,
          cycles,
          G_weight_path,
          D_weight_path,
          result_path)