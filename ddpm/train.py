import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets import (GaussianDiffusion, UNet, generate_cosine_schedule,
                  generate_linear_schedule)
from utils.callbacks import LossHistory
from utils.dataloader import Diffusion_dataset_collate, DiffusionDataset
from utils.utils import get_lr_scheduler, set_optimizer_lr, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #-------------------------------#
    #   Whether to use Cuda
    #   Set to False if GPU is not available
    #-------------------------------#
    Cuda            = True
    #---------------------------------------------------------------------#
    #   distributed     Specifies whether to use single-machine multi-GPU distributed training
    #                   Terminal commands only support Ubuntu. CUDA_VISIBLE_DEVICES is used to specify the GPU on Ubuntu.
    #                   On Windows, DP mode is used by default to call all GPUs, and DDP is not supported.
    #   DP mode:
    #       Set             distributed = False
    #       In terminal:    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Set             distributed = True
    #       In terminal:    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #distributed     = True
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #               Can reduce memory usage by about half, requires pytorch 1.7.1 or higher
    #---------------------------------------------------------------------#
    fp16            = False
    #fp16            = True
    #---------------------------------------------------------------------#
    #   If you want to resume training, set model_path to the weights file already trained in the logs folder.
    #   When model_path = '', the entire model's weights are not loaded.
    #
    #   Here, the entire model's weights are used, so they are loaded in train.py.
    #   If you want to train the model from scratch, set model_path = ''.
    #---------------------------------------------------------------------#
    diffusion_model_path    = ""
    #---------------------------------------------------------------------#
    #   Setting for convolution channels, can reduce if memory is insufficient, e.g., to 64
    #---------------------------------------------------------------------#
    #channel         = 128
    channel         = 64
    #---------------------------------------------------------------------#
    #   Parameters related to betas
    #---------------------------------------------------------------------#
    schedule        = "linear"
    #num_timesteps   = 1000
    num_timesteps   = 800 #400
    print('num_timesteps', num_timesteps)
    schedule_low    = 1e-4
    schedule_high   = 0.02
    #---------------------------------------------------------------------#
    #   Setting for image size, e.g., [128, 128]
    #   After setting, the Diffusion images cannot be seen during training and need to be viewed in single images during prediction.
    #---------------------------------------------------------------------#
    #input_shape     = (32, 32)
    input_shape     = (9, 9)
    
    #------------------------------#
    #   Training parameter settings
    #------------------------------#
    Init_Epoch      = 0
    #Epoch           = 1000
    Epoch           = 600
    #batch_size      = 64
    batch_size      = 256     # 512*2, 2048
    
    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate decay, etc.
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         Maximum learning rate of the model
    #   Min_lr          Minimum learning rate of the model, default is 0.01 times the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-4   #2e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  The type of optimizer used, options are adam, adamw
    #   momentum        Momentum parameter used in the optimizer
    #   weight_decay    Weight decay, can prevent overfitting
    #                   Using adam can lead to incorrect weight decay, recommended to set to 0 when using adam.
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   The type of learning rate decay used, options are step, cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     Save weights every number of epochs
    #------------------------------------------------------------------#
    save_period         = 50 # original with 50
    #------------------------------------------------------------------#
    #   save_dir        Folder to save weights and log files
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   num_workers     Set whether to use multi-threading for data loading
    #                   Enabling this will speed up data loading but use more memory
    #                   Computers with limited memory can set this to 2 or 0
    #------------------------------------------------------------------#
    num_workers         = 4
    #------------------------------------------#
    #   Get image paths
    #------------------------------------------#
    annotation_path = "train_lines.txt"

    import pickle
    pickle_file = 'sudoku_np.pkl'
    with open(pickle_file, 'rb') as file:
        loaded_array = pickle.load(file)

    print("Pickle array loaded from file:",)
    sudoku_array= np.transpose(loaded_array,(0, 3, 1, 2)) # change shape to the normal network channel order
    print(sudoku_array.shape) 
    sudoku_array_shape= sudoku_array.shape # (1000000, 9, 9, 9)
    
    lines= sudoku_array_shape
    #------------------------------------------------------#
    #   Set the GPU used
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0

    if schedule == "cosine":
        betas = generate_cosine_schedule(num_timesteps)
    else:
        betas = generate_linear_schedule(
            num_timesteps,
            schedule_low * 1000 / num_timesteps,
            schedule_high * 1000 / num_timesteps,
        )
    #------------------------------------------#
    #   Diffusion network
    #------------------------------------------#
    #diffusion_model = GaussianDiffusion(UNet(3, channel), input_shape, 3, betas=betas)
    diffusion_model = GaussianDiffusion(UNet(9, channel), input_shape, 9, betas=betas)
    total_params = sum(p.numel() for p in diffusion_model.parameters())
    print(f"Total number of parameters: {total_params}")

    #------------------------------------------#
    #   Reload the trained model
    #------------------------------------------#
    if diffusion_model_path != '':
        model_dict      = diffusion_model.state_dict()
        pretrained_dict = torch.load(diffusion_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        diffusion_model.load_state_dict(model_dict)

    #----------------------#
    #   Record Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, [diffusion_model], input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    #   torch 1.2 does not support amp, recommend using torch 1.7.1 or above for correct fp16 usage
    #   Hence torch1.2 shows "could not be resolved" here
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    
    diffusion_model_train = diffusion_model.train()
    
    if Cuda:
        if distributed:
            diffusion_model_train = diffusion_model_train.cuda(local_rank)
            diffusion_model_train = torch.nn.parallel.DistributedDataParallel(diffusion_model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            cudnn.benchmark = True
            diffusion_model_train = torch.nn.DataParallel(diffusion_model)
            diffusion_model_train = diffusion_model_train.cuda()

    # with open(annotation_path) as f:
    #     lines = f.readlines()  # here lines are the paths list in the picture data folder
    #num_train = len(lines)      # here shows the length of them 
    num_train = sudoku_array_shape[0] # 1000000 in (1000000, 9, 9, 9)

    if local_rank == 0:
        show_config(
            input_shape = input_shape, Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train
            )
    #------------------------------------------------------#
    #   Init_Epoch is the starting epoch
    #   Epoch is the total number of training epochs
    #------------------------------------------------------#
    if True:
        #---------------------------------------#
        #   Select optimizer based on optimizer_type
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(diffusion_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay = weight_decay),
            'adamw' : optim.AdamW(diffusion_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay = weight_decay),
        }[optimizer_type]
        
        #---------------------------------------#
        #   Get learning rate decay formula
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)
        
        #---------------------------------------#
        #   Determine the length of each epoch
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        if epoch_step == 0:
            raise ValueError("Dataset too small for training, please expand the dataset.")

        #---------------------------------------#
        #   Build dataset loader
        #---------------------------------------#
        train_dataset   = DiffusionDataset(sudoku_array, lines, input_shape)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            shuffle         = True
    
        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=Diffusion_dataset_collate, sampler=train_sampler)

        #---------------------------------------#
        #   Start training the model
        #---------------------------------------#
        for epoch in range(Init_Epoch, Epoch):

            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(diffusion_model_train, diffusion_model, loss_history, optimizer, 
                        epoch, epoch_step, gen, Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
            
            # # Save model weights
            # if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            #     save_path = os.path.join(save_dir, f"epoch_{epoch + 1}.pth")
            #     torch.save(diffusion_model.state_dict(), save_path)
            #     print(f"Saved model at epoch {epoch + 1} to {save_path}")

            if distributed:
                dist.barrier()

