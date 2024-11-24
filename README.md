# Topic of Masterarbeit_FAU_HHI: Discrete Data Generation with Trajectory Models

## Contents
The current code consists of two parts: modified DDSM and modified DDPM.

## Description 
### How to set up the environment
Use `base_env_pytorch.def` to set up the environment to get a `.sif` file with the same name.

### How to run modified DDSM
1. Run `MA_2024_prepsamples.sh` to get a simulated diffusion path. In this `.sh` file, the `presample_noise.py` will be called. There are several parameters which can be set up:
    - `-n`, `--num_samples`, type=int
    - `-c`, `--num_cat`, type=int
    - `-t`, `--num_time_steps`, type=int
    - `--speed_balance`, type=bool, default=false
    - `--max_time`, type=float
    - `--out_path`, type=str
    - `--boundary_mode`, choices=['clamp', 'reflect_boundaries', 'reflection'], default='clamp'

    Example:
    ```bash
    presample_noise.py -n 50000 -c 9 -t 400 --max_time 1 --out_path sudoku/ --boundary_mode 'reflect_boundaries'
    ```

2. Run `MA_2024_results_ddsm_channel_std.sh` to check the quality of path simulation end distribution.

3. Run `MA_2024_train.sh` to train the model. The `.pth` files are saved after each 50 epochs. There are several parameters which can be set up:
    - `-t`, `--num_time_steps`, type=int, default=400
    - `-spd`, `--speed_balance`, choices=['s1', 'sab'], default='s1'
    - `-bm`, `--boundary_mode`, choices=['clamp', 'reflect_boundaries', 'reflection'], default='clamp'

    Example:
    ```bash
    train_sudoku_allinone.py -t 100 -spd 's1' -bm 'clamp'
    ```

4. Run `MA_2024_eval_Epoch.sh` to evaluate the trained model.

### How to run modified DDPM
1. Run `SH_sudoku_csv_to_pickle.sh` to extract Sudoku data from a pre-randomly created Sudoku `.csv` file, or create random Sudoku during the training phase. (Preferably, extract data from a `.csv` file to save training time.)

2. Run `SH_ddpm_train.sh` to train the model. You can change the UNet size in `./ddpm/nets/unet.py`.

3. Run `SH_ddpm_predict.sh` to evaluate the trained model.

### Environment requirements
- Python 3.8+
- CUDA: 11.7.1-cudnn8-devel-ubuntu20.04