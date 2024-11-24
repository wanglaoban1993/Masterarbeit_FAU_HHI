import itertools
import math
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import os


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB;

# Convert the image to RGB to prevent grayscale images from reporting errors during prediction.
# The code only supports prediction of RGB images, all other types of images will be converted to RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#----------------------------------------#
#   预处理训练图片
# Preprocess training images
#----------------------------------------#
def preprocess_input(x):
    x /= 255
    x -= 0.5
    x /= 0.5
    return x

def postprocess_output(x):
    x *= 0.5
    x += 0.5
    x *= 255
    return x

# def show_result(num_epoch, net, device):
#     test_images = net.sample(4, device)

#     size_figure_grid = 2
#     fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
#     for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
#         ax[i, j].get_xaxis().set_visible(False)
#         ax[i, j].get_yaxis().set_visible(False)
#     for k in range(2 * 2):
#         i = k // 2
#         j = k % 2
#         ax[i, j].cla()
#         ax[i, j].imshow(np.uint8(postprocess_output(test_images[k].cpu().data.numpy().transpose(1, 2, 0))))

#     label = 'Epoch {0}'.format(num_epoch)
#     fig.text(0.5, 0.04, label, ha='center')
#     plt.savefig("results/train_out/epoch_" + str(num_epoch) + "_results.png")
#     plt.close('all') 

def show_result(num_epoch, net, device):
    # 使用模型生成测试图像
    # Generate test images using the model
    test_images = net.sample(4, device)  # 生成的图像，形状为 [4, 9, 16, 16]; the generative image with shape[4, 9, 16, 16]

    # 创建保存目录; Create a save directory
    save_dir = "results/train_out"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 循环遍历生成的图像并保存; Loop through the generated images and save them
    for idx in range(4):  # 遍历4张生成的图像; Traverse the 4 generated images
        # 获取单张图像数据; Traverse the one generated image
        #image_data = test_images[idx].cpu().data.numpy().transpose(1, 2, 0)  # with shape [16, 16, 9]
        image_data = test_images[idx].cpu().data.numpy()
        print('created image_data.shape: after one epoch training', image_data.shape)

        # 保存图像数据为 `.npy` 文件（NumPy格式）; Save image data as `.npy` file (NumPy format)
        save_path = os.path.join(save_dir, f"epoch_{num_epoch}_image_{idx}.npy")
        np.save(save_path, image_data) 

        # 如果需要将图像保存为常见格式（如 PNG），可选择保存单个通道或进行通道合并
        # 例如，仅保存第一个通道为灰度图：
	# If you need to save the image in a common format (such as PNG), you can choose to save a single channel or merge channels
	# For example, save only the first channel as grayscale:
        
        # from PIL import Image
        # single_channel_image = Image.fromarray(np.uint8(image_data[:, :, 0] * 255))  # 取第一个通道; get the first channel
        # single_channel_image.save(os.path.join(save_dir, f"epoch_{num_epoch}_image_{idx}.png"))

    print(f"Saved generated data for epoch {num_epoch}.")
    
# import numpy as np
# image_data = np.load("results/train_out/epoch_1_image_0.npy")
# print(image_data.shape)  # 输出: (16, 16, 9)

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

#---------------------------------------------------#
#   获得学习率; Get the learning rate;
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
