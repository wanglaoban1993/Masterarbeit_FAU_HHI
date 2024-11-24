import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


# class DiffusionDataset(Dataset):
#     def __init__(self, annotation_lines, input_shape):
#         super(DiffusionDataset, self).__init__()

#         self.annotation_lines   = annotation_lines
#         self.length             = len(annotation_lines)
#         self.input_shape        = input_shape

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         image   = Image.open(self.annotation_lines[index].split()[0])
#         image   = cvtColor(image).resize([self.input_shape[1], self.input_shape[0]], Image.BICUBIC)
        
#         image   = np.array(image, dtype=np.float32)
#         image   = np.transpose(preprocess_input(image), (2, 0, 1))
#         return image
class DiffusionDataset(Dataset):    ##### in this dataset: i read the created 1000000 correct flatnted sudoku, then switch it into one hot coding,
                                    ##### then add 0 to the added empty channel to fit the 16*16 shape of the original network
    def __init__(self, sudoku_array, sudoku_array_shape, input_shape):
        super(DiffusionDataset, self).__init__()
        
        self.sudoku_array         = sudoku_array
        self.sudoku_array_shape   = sudoku_array_shape
        self.length               = sudoku_array_shape[0]
        self.input_shape          = input_shape

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #image   = Image.open(self.annotation_lines[index].split()[0])
        #image   = cvtColor(image).resize([self.input_shape[1], self.input_shape[0]], Image.BICUBIC)
        image   = self.sudoku_array[index]
        #new_dataset = np.zeros((self.input_shape[0], self.input_shape[1], 9), dtype=np.float32)
        #new_dataset[3:12, 3:12, :] = image
        #image_new = np.transpose(new_dataset, (2, 0, 1)) # change one hot coding channel to the first(9, 16, 16), others keep the same oder to match the original network
        #image   = np.transpose(preprocess_input(image), (2, 0, 1))
        #return image_new
        return image

def Diffusion_dataset_collate(batch):
    images = []
    for image in batch:
        images.append(image)
    images = torch.from_numpy(np.array(images, np.float32))
    return images
