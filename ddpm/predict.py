#-------------------------------------#
import numpy as np
from ddpm import Diffusion, sudoku_acc

if __name__ == "__main__":
    # save_path_5x5 = "results/predict_out/predict_5x5_results.png"
    # save_path_1x1 = "results/predict_out/predict_1x1_results.png"
    #save_path_sudoku= 
    generate_batch_size= 8

    ddpm = Diffusion()
    # while True:
        #img = input('Just Click Enter~')
        # print("Generate_1x1_image")
        # ddpm.generate_1x1_image(save_path_1x1)
        # print("Generate_1x1_image Done")
        
        # print("Generate_5x5_image")
        # ddpm.generate_5x5_image(save_path_5x5)
        # print("Generate_5x5_image Done")
    print(f'creatting sudoku with batch size: {generate_batch_size}*1000')
    sudoku_batch= ddpm.generate_batch_sudoku(generate_batch_size) # creat (batch_size, 9(one hot coding), 9, 9)
    print('created sudoku according to the trained model finished')
    print('sudoku_batch', sudoku_batch.shape)

    cutted_sudoku_batch= np.transpose(sudoku_batch, (0, 2, 3, 1)) # change back to shape(batch_size, width, heigth, one_hot_encoded)
    print('cutted_sudoku_batch', cutted_sudoku_batch.shape)

    corret_rate= sudoku_acc(cutted_sudoku_batch)
    print('sudoku corret rate conculated')
        
