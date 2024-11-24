import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
import torch

from nets import (GaussianDiffusion, UNet, generate_cosine_schedule,
                  generate_linear_schedule)
from utils.utils import postprocess_output, show_config


class Diffusion(object):
    _defaults = {
        #-----------------------------------------------#
        #   model_path points to the weight file in the logs folder
        #-----------------------------------------------#
        #"model_path"        : './logs/loss_2024_08_31_18_03_21/Diffusion_Epoch50-GLoss0.0172.pth',  #'model_data/Diffusion_Flower.pth', 
        #"model_path"        : './logs/loss_2024_09_01_12_50_47/Diffusion_Epoch200-GLoss0.0147.pth', 
        
        #"model_path"        : './logs/Diffusion_Epoch600-GLoss0.0161.pth',
        "model_path"        : './logs/loss_2024_08_31_18_03_21/Diffusion_Epoch600-GLoss0.0140.pth',    
        #-----------------------------------------------#
        #   Setting for convolution channels
        #-----------------------------------------------#
        #"channel"           : 64,                       
        "channel"           : 128,                     
        #-----------------------------------------------#
        #   Setting for input image size
        #-----------------------------------------------#
        #"input_shape"       : (32, 32),
        "input_shape"       : (9, 9),
        #-----------------------------------------------#
        #   Parameters related to betas
        #-----------------------------------------------#
        "schedule"          : "linear",
        #"num_timesteps"     : 800,
        "num_timesteps"     : 400,
        "schedule_low"      : 1e-4,
        "schedule_high"     : 0.02,
        #-------------------------------#
        #   Whether to use Cuda
        #   Set to False if GPU is not available
        #-------------------------------#
        "cuda"              : True,
        #"generate_batch_size": 256
    }

    #---------------------------------------------------#
    #   Initialize Diffusion
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  
            self._defaults[name] = value 
        self.generate()

        show_config(**self._defaults)

    def generate(self):
        #----------------------------------------#
        #   Create Diffusion model
        #----------------------------------------#
        if self.schedule == "cosine":
            betas = generate_cosine_schedule(self.num_timesteps)
        else:                         # default we use linear here
            betas = generate_linear_schedule(
                self.num_timesteps,
                self.schedule_low * 1000 / self.num_timesteps,
                self.schedule_high * 1000 / self.num_timesteps,
            )
            
        # self.net    = GaussianDiffusion(UNet(3, self.channel), self.input_shape, 3, betas=betas)
        self.net    = GaussianDiffusion(UNet(9, self.channel), self.input_shape, 9, betas=betas) # change in channel as 9
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = self.net.cuda()

    #---------------------------------------------------#
    #   Generate 5x5 images using Diffusion
    #---------------------------------------------------#
    def generate_5x5_image(self, save_path):
        with torch.no_grad():
            randn_in    = torch.randn((1, 1)).cuda() if self.cuda else torch.randn((1, 1))

            test_images = self.net.sample(25, randn_in.device)

            size_figure_grid = 5
            fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
            for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

            for k in range(5*5):
                i = k // 5
                j = k % 5
                ax[i, j].cla()
                ax[i, j].imshow(np.uint8(postprocess_output(test_images[k].cpu().data.numpy().transpose(1, 2, 0))))

            label = 'predict_5x5_results'
            fig.text(0.5, 0.04, label, ha='center')
            plt.savefig(save_path)

    #---------------------------------------------------#
    #   Generate 1x1 image using Diffusion
    #---------------------------------------------------#
    def generate_1x1_image(self, save_path):
        with torch.no_grad():
            randn_in    = torch.randn((1, 1)).cuda() if self.cuda else torch.randn((1, 1))

            test_images = self.net.sample(1, randn_in.device, use_ema=False)
            test_images = postprocess_output(test_images[0].cpu().data.numpy().transpose(1, 2, 0))

            Image.fromarray(np.uint8(test_images)).save(save_path)
    
    #---------------------------------------------------#
    #   Build batch of Sudoku and check the accuracy 
    #---------------------------------------------------#
    def generate_batch_sudoku(self, generate_batch_size):
        with torch.no_grad():
            num_images= generate_batch_size*1000
            randn_in = torch.randn((num_images, 9, 9, 9)).cuda() if self.cuda else torch.randn((num_images, 9, 9, 9))   # Generate random noise for 256 images
            generate_sudoku= self.net.sample(num_images, randn_in.device, use_ema=False)                                # Generate 256*1000 images
            return generate_sudoku

def sudoku_acc(sample, return_array=False):
    sample = sample.detach().cpu().numpy()
    correct = 0
    total = sample.shape[0]
    ans = sample.argmax(-1) + 1
    numbers_1_N = np.arange(1, 9 + 1)
    corrects = []
    for board in ans:
        if (np.all(np.sort(board, axis=1) == numbers_1_N) and
                np.all(np.sort(board.T, axis=1) == numbers_1_N)):
            # Check blocks

            blocks = board.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
            if np.all(np.sort(board.T, axis=1) == numbers_1_N):
                correct += 1
                corrects.append(True)
            else:
                corrects.append(False)
        else:
            corrects.append(False)

    if return_array:
        return corrects
    else:
        print('i start here')
        correct_rate= 100 * correct / total
        print('correct {} %'.format(correct_rate))
        return correct_rate

