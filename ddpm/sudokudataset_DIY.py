import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import numpy as np


file_path = 'sudoku.csv'  
data = pd.read_csv(file_path)


print(data.head())
print(data.info())
print(data['solutions'])

sudoku_cut= data['solutions']
print(sudoku_cut.shape, type(sudoku_cut), type(sudoku_cut[0]))

import numpy as np
import pandas as pd

sudoku_cut = data['solutions']

# switch str to int array and restore to original 9*9 format
sudoku_matrices = np.array([np.array(list(map(int, list(sudoku)))).reshape(9, 9) for sudoku in sudoku_cut])

# change to one hot coding style
one_hot_encoded = np.eye(9)[sudoku_matrices - 1] # the created last demension is one hot coding!!!

# save numpy file
with open('sudoku_np.pkl', 'wb') as f:
    pickle.dump(one_hot_encoded, f)

print("One-hot encoded and saved as sudoku_np.pkl")
