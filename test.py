import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from generator import G_model
from discriminator import D_model
from data import dataset
from PIL import Image
import sys
import os
import gc 
from tqdm import tqdm
from skimage import measure
import cv2
import copy
import time

def thres(label):
    thresh = 0.5 # Post-processing threshold
    thresh_pred = label
    thresh_pred[thresh_pred < thresh] = 0 # post-processing, value less than threshold is set to 0
    thresh_pred[thresh_pred > (thresh)] = 1 # post-processing, value greater than threshold is set to 1
  
    return thresh_pred

def test_all(model,D, device, result_path):
    # Evaluate the model against test data
    model.to(device)
    D.to(device)
    model.eval()
    D.eval()

    # Test dataset
    test_data_obj = dataset(test=True) # BnG dataset 
    test_data_loader = DataLoader(test_data_obj, batch_size=4, shuffle=False)
    times=[]
    # Do not need to test label (only predict on test dataset)
    for data, label, img_name in tqdm(test_data_loader):
        data = data.to(device)
        label = label.to(device)
        start_time=time.time()
        predictions, init_guesses = model(data)
        predictions_D=D(predictions)

        # Display prediction result
        for i in range(len(predictions)):
            pred = predictions[i].permute(1,2,0).cpu().detach().numpy()
            pred_D=predictions_D[i].permute(1,2,0).cpu().detach().numpy()
            init_guess = init_guesses[i].permute(1,2,0).cpu().detach().numpy()
            img = data[i].permute(1,2,0).cpu().detach().numpy()
            orig_img = Image.fromarray((img * 255).astype(np.uint8))
            
            # Predict all test images
            thresh = 0.5 # Post-processing threshold
            thresh_pred = pred
            thresh_pred[thresh_pred < thresh] = 0 # post-processing, value less than threshold is set to 0
            thresh_pred[thresh_pred > (thresh)] = 1 # post-processing, value greater than threshold is set to 1

            regions, num = measure.label(thresh_pred, return_num=True, background=1, connectivity=2)

            region_sizes = np.bincount(regions.ravel())

            region_copy = copy.deepcopy(region_sizes)
            region_copy.sort()
            top_3 = region_copy[-3] # Get the largest three pixel groups
            not_top = region_copy[:-2]

            # Check the three pixel groups and change the threshold if the regions are insignificant
            if(len(region_copy) > 3 and region_copy[-1] > 10000 and region_copy[-2] > 10000 and region_copy[-3] < 1000):
                top_3 = region_copy[-2]

            mask = np.zeros_like(thresh_pred)

            # Go through each region and eliminate small and scattered regions
            for region_index, region_size in enumerate(region_sizes):
                if region_size < top_3:
                    mask[regions == region_index] = 1

            filtered_segmented_image = cv2.bitwise_or(mask, thresh_pred)

            erode_kernel = np.ones((6, 6), np.uint8)
            dialate_kernel = np.ones((6, 6), np.uint8)
            opening_kernel = np.ones((2, 2), np.uint8)

            erode_image = cv2.erode(filtered_segmented_image, dialate_kernel)
            erode_image_pred = thres(erode_image)

            dilate_image = cv2.dilate(erode_image_pred, erode_kernel)
            dilate_image_pred = thres(dilate_image)

            opening = cv2.morphologyEx(erode_image_pred, cv2.MORPH_OPEN, opening_kernel)
            opening=opening.reshape((opening.shape[0], opening.shape[1],1))
            # save mask (black and white)
            
            if not os.path.exists(result_path+'predicted_masks/'):
                os.makedirs(result_path+'predicted_masks/')
            proc_black_pred = black_rgb_label(opening) # convert to 3 channels
            proc_black_label = Image.fromarray((proc_black_pred * 255).astype(np.uint8))
            proc_black_label.save(result_path+f'predicted_masks/{img_name[i]}')
            
            # save init_guess result
            if not os.path.exists(result_path+'init_guess/'):
                os.makedirs(result_path+'init_guess/')
            proc_black_init_guess = black_rgb_label_init_guess(init_guess) # convert to 3 channels
            proc_black_label = Image.fromarray((proc_black_init_guess * 255).astype(np.uint8))
            proc_black_label.save(result_path+f'init_guess/{img_name[i]}')

            # save superimposed image (prediction in red)
            if not os.path.exists(result_path+'superimposed_new/'):
                os.makedirs(result_path+'superimposed_new/')
               
            proc_red_pred = rgb_label(opening) # convert to rgb (red)
            red_label = Image.fromarray((proc_red_pred * 255).astype(np.uint8))
            imposed_red_img = Image.blend(orig_img, red_label, alpha=0.3)
            imposed_red_img.save(result_path+f'superimposed_new/{img_name[i]}')

# convert label to rgb
def rgb_label(label):
	pred_rgb = np.ones((label.shape[0], label.shape[1], 3))
	for h in range(label.shape[0]):
		for w in range(label.shape[1]):
			pred_rgb[h][w][0] = 1 - label[h][w][0]
			pred_rgb[h][w][1] = 0
			pred_rgb[h][w][2] = 0
	return pred_rgb


# convert grayscale to 3 channels
def black_rgb_label(label):
	pred_rgb = np.ones((label.shape[0], label.shape[1], 3))
	for h in range(label.shape[0]):
		for w in range(label.shape[1]):
			pred_rgb[h][w][0] = label[h][w][0]
			pred_rgb[h][w][1] = label[h][w][0]
			pred_rgb[h][w][2] = label[h][w][0]
	return pred_rgb


def black_rgb_label_init_guess(label):
	pred_rgb = np.ones((label.shape[0], label.shape[1], 3))
	for h in range(label.shape[0]):
		for w in range(label.shape[1]):
			pred_rgb[h][w][0] = label[h][w][0]
			pred_rgb[h][w][1] = label[h][w][0]
			pred_rgb[h][w][2] = label[h][w][0]
	return pred_rgb


if __name__ == "__main__":
    batch_size = 8 
    is_red_pred = True
    cuda = "cuda:0"
    
    device = torch.device(cuda)
    
    G = G_model(device=device)
    D = D_model(device=device)
    
    G.load_state_dict(torch.load(''))
    D.load_state_dict(torch.load(''))
    
    result_path=''
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
test_all(G, D, device, result_path)