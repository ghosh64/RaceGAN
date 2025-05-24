import torch.nn as nn
import torch

class WBA_model(nn.Module):
	def __init__(self, device):
		super().__init__()

		self.device = device 

		# Branch 1: (larger stride/pool size/filter size) (General Feature)
		self.b1_conv1 = nn.Sequential(
						nn.Conv2d(1, 64, kernel_size=3, stride=3),
						nn.LeakyReLU(),
						nn.AvgPool2d(kernel_size=3, stride=3),
						nn.Dropout2d(),
						nn.BatchNorm2d(64),
						nn.Upsample(size=128)                      
						)

		self.b1_conv2 = nn.Sequential(
						nn.Conv2d(64, 256, kernel_size=3, stride=3),
						nn.LeakyReLU(),
						nn.AvgPool2d(kernel_size=3, stride=3),
						nn.BatchNorm2d(256)
						)

		self.b1_resize = nn.Sequential(
							nn.ConvTranspose2d(256, 64, kernel_size=3, padding=(2,2)),
							nn.Upsample(size=64),
							nn.ConvTranspose2d(64, 1, kernel_size=3, padding=(2,2)),
							nn.Upsample(size=320)
						)

		# Branch 2: local, smaller convolution kernels, or even 1x1 convolutions
		self.b2_conv1 = nn.Sequential(
						nn.Conv2d(1, 128, kernel_size=2, stride=1),
						nn.LeakyReLU(),
						nn.AvgPool2d(kernel_size=2, stride=1),
						nn.BatchNorm2d(128)
						)

		self.b2_conv2 = nn.Sequential(
						nn.Conv2d(128, 512, kernel_size=2, stride=2, padding=(2,2)),
						nn.LeakyReLU(),
						nn.AvgPool2d(kernel_size=2, stride=1),
						nn.BatchNorm2d(512)
						)

		self.b2_resize = nn.Sequential(
						nn.ConvTranspose2d(512, 128, kernel_size=3, stride=3, padding=(3,3)),
						nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2, padding=(2,2)), # only 1 feature channel --> grey scale
						nn.Upsample(320)
						)

		# Final smoothen layer
		self.fin_conv = nn.Sequential(
						nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=1),
						nn.ConvTranspose2d(1, 1, kernel_size=4, stride=1, padding=(2,2)), 
						nn.BatchNorm2d(1) 
						)

		self.sig = nn.Sigmoid()

	def forward(self,x):
		# Branch 1
		x1 = self.b1_conv1(x)
		x1 = self.b1_conv2(x1)
		x1 = self.b1_resize(x1)

		# Branch 2
		# Crop image (1/6 from top)
		height = x.shape[2]
		width = x.shape[3]
		x2 = x[:,:,height//6:,:] # [8, 3, 267, 320]

		x2 = self.b2_conv1(x)
		x2 = self.b2_conv2(x2)

		# fill cropped part with 0
		fill_tensor = torch.zeros((x2.shape[0], x2.shape[1], (x2.shape[3]-x2.shape[2]), x2.shape[3]))
		fill_tensor = fill_tensor.to(self.device) # convert tensor from cpu to gpu
		x2 = torch.cat((fill_tensor, x2), 2) # concatenate top with 0

		x2 = self.b2_resize(x2)

		# Crop x1 and x2 into top, middle, bottom
		x1_top = x1[:,:,:80,:] # 0 ~ 79 (top 1/4)
		x1_mid = x1[:,:,80:160,:] # middle 1/4
		x1_bot = x1[:,:,160:,:] # bottom 1/2

		x2_top = x2[:,:,:80,:] # 0 ~ 79 (top 1/4)
		x2_mid = x2[:,:,80:160,:] # middle 1/4
		x2_bot = x2[:,:,160:,:] # bottom 1/2

		# Apply weights to each branch and add them up
		x_top = 0.7 * x1_top + 0.3 * x2_top # global feature heavy (7:3)
		x_mid = 0.4 * x1_mid + 0.6 * x2_mid # 4:6
		x_bot = 0.3 * x1_bot + 0.7 * x2_bot # local feature heavy (3:7)

		# Concatenate top, mid, and bottom
		x = torch.cat((x_top, x_mid, x_bot), 2)

		# Smoothen convolutional layer
		x = self.fin_conv(x)

		return x