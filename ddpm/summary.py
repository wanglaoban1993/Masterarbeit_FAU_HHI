#--------------------------------------------#
#   to check paramters of the network
#--------------------------------------------#
import torch
from thop import clever_format, profile

from nets.unet import UNet

if __name__ == '__main__':
    #input_shape         = [64, 64]
    input_shape         = [9, 9]
    num_timesteps       = 1000
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #m       = UNet(3)
    m       = UNet(9)
    for i in m.children():
        print(i)
        print('==============================')
        
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    t               = torch.randint(0, num_timesteps, (1,), device=device)
    flops, params   = profile(m.to(device), (dummy_input, t), verbose=False)
    #--------------------------------------------------------#
    # flops * 2 because the profile does not consider convolution as two operations
    # Some papers consider convolution as two operations, multiplication and addition. In this case, multiply by 2
    # Some papers only consider the number of multiplication operations and ignore addition. In this case, do not multiply by 2
    # This code chooses to multiply by 2, refer to YOLOX.
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
