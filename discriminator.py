# discriminator.py
import torch
import torch.nn as nn
import torchvision.transforms.functional as func
from WBA import WBA_model

class D_model(nn.Module):
	'''
	Same model from generator
	Input: mask (1 x 320 x 320)
	Output: tensor with same dimension as input with values from 0 to 1
			0: fake mask, 1: real mask
	'''
	def __init__(self, device):
		super(D_model, self).__init__()
		self.device = device
		self.WBA_model = WBA_model(self.device).to(device)


		self.max_1 = nn.MaxPool2d(kernel_size=10, stride=10)
		self.max_2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.bi_up = nn.UpsamplingBilinear2d(size=320)

		self.conv_1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=4, stride=3, padding=(3,3), padding_mode="reflect"),
			nn.ReLU(),
			nn.InstanceNorm2d(32),
		)

		self.conv_2 = nn.Sequential(
			nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=(2,2), padding_mode="reflect"),
			nn.ReLU(),
			nn.InstanceNorm2d(128),
		)

		self.conv_3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=2, stride=2),
			nn.ReLU(),
			nn.InstanceNorm2d(256),
		)

		self.t_conv_1 = nn.Sequential(
			nn.ConvTranspose2d(256, 64, kernel_size=4, stride=3), # [64, 82, 82]
			nn.UpsamplingBilinear2d(160),
			nn.MaxPool2d(kernel_size=2, stride=2), # [64, 80, 80]
			nn.ReLU(),
		)

		self.t_conv_2 = nn.Sequential(
			nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2), # [16, 161, 161]
			nn.UpsamplingBilinear2d(320),
			nn.MaxPool2d(kernel_size=2, stride=2), # [16, 160, 160]
			nn.ReLU(),
		)

		self.t_conv_3 = nn.Sequential(
			nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2), # [16, 320, 320]
			nn.UpsamplingBilinear2d(640),
			nn.MaxPool2d(kernel_size=2, stride=2), # [16, 320, 320]
			nn.ReLU(),
			nn.BatchNorm2d(1),
		)

	def forward(self, x):
		# initial processing
		mask = self.max_1(x) # first max pooling
		mask = self.max_2(mask) # second max pooling
		mask = self.classify_pix(img=mask, scale=0.15)
		mask = self.bi_up(mask)
		mask = self.conv_1(mask)
		mask = self.conv_2(mask) 
		mask = self.conv_3(mask) 
		mask = self.t_conv_1(mask)
		mask = self.t_conv_2(mask)
		mask = self.t_conv_3(mask)

		return mask

	def classify_pix(self, img:torch.Tensor, scale:float) -> torch.Tensor:
		'''
		change each pixel's value
		less than average -> multiplied by (1 - scale)
		graeter than average -> multiplied by (1 + scale)
		scale: desired pixel value percentage to adjust
		'''
		thresh = torch.mean(img, dtype=torch.float)
		img[img < thresh] *= (1-scale)
		img[img >= thresh] *= (1+scale)
		return img