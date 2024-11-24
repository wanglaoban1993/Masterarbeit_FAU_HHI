import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLU(nn.Module):  
    # SiLU activation function
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")
    
#------------------------------------------#
#   Calculate positional embedding for time steps.
#   Half is sin, half is cos.
#------------------------------------------#
class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device      = x.device
        half_dim    = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Outer product of x * self.scale and emb
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

#------------------------------------------#
#   Downsampling layer, a convolution with stride 2x2
#------------------------------------------#
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x, time_emb, y):
        return self.downsample(x)

#------------------------------------------#
#   Upsampling layer, Upsample + convolution
#------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
        
    def forward(self, x, time_emb, y):
        return self.upsample(x)

#------------------------------------------#
#   Use Self-Attention mechanism
#   Apply global Self-Attention
#------------------------------------------#
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w  = x.shape
        q, k, v     = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention   = torch.softmax(dot_products, dim=-1)
        out         = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out         = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out) + x
    
#------------------------------------------#
#   Residual block for feature extraction
#------------------------------------------#
class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, dropout, time_emb_dim=None, num_classes=None, activation=SiLU(),
        norm="gn", num_groups=32, use_attention=False,
    ):
        super().__init__()

        self.activation = activation

        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout), 
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.time_bias  = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

        self.residual_connection    = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attention              = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups)
    
    def forward(self, x, time_emb=None, y=None):
        out = self.activation(self.norm_1(x))
        # First convolution
        out = self.conv_1(out)
        
        # Apply a fully connected layer to time_emb, acting on channels
        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]

        # Apply a fully connected layer to y_emb, acting on channels
        if self.class_bias is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")

            out += self.class_bias(y)[:, :, None, None]

        out = self.activation(self.norm_2(out))
        # Second convolution + residual connection
        out = self.conv_2(out) + self.residual_connection(x)
        # Finally apply Attention
        out = self.attention(out)
        return out

#------------------------------------------#
#   Unet model
#------------------------------------------#
class UNet(nn.Module):
    def __init__(
        self, img_channels, base_channels=128, channel_mults=(1, 2, 4, 8),
        num_res_blocks=3, time_emb_dim=128 * 4, time_emb_scale=1.0, num_classes=None, activation=SiLU(),
        dropout=0.1, attention_resolutions=(1,), norm="gn", num_groups=32, initial_pad=0,
    ):
        super().__init__()
        # Activation function used, generally SILU
        self.activation = activation
        # Whether to apply padding to the input
        self.initial_pad = initial_pad
        # Number of classes to distinguish
        self.num_classes = num_classes
        
        # Fully connected layer for time input
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None
    
        # First convolution for input image
        self.init_conv  = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        # self.downs is used to store layers for downsampling. First, use ResidualBlock for feature extraction.
        # Then use Downsample to reduce the height and width of the feature map.
        self.downs      = nn.ModuleList()
        self.ups        = nn.ModuleList()
        
        # channels refers to the number of channels processed by each module
        # now_channels is an intermediate variable representing the current number of channels
        channels        = [base_channels]
        now_channels    = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(
                    ResidualBlock(
                        now_channels, out_channels, dropout,
                        time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation,
                        norm=norm, num_groups=num_groups, use_attention=i in attention_resolutions,
                    )
                )
                now_channels = out_channels
                channels.append(now_channels)
            
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)

        # Can be seen as feature integration, a feature extraction module in the middle
        self.mid = nn.ModuleList(
            [
                ResidualBlock(
                    now_channels, now_channels, dropout,
                    time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation,
                    norm=norm, num_groups=num_groups, use_attention=True,
                ),
                ResidualBlock(
                    now_channels, now_channels, dropout,
                    time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation, 
                    norm=norm, num_groups=num_groups, use_attention=False,
                ),
            ]
        )

        # Perform upsampling for feature fusion
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels, out_channels, dropout, 
                    time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation, 
                    norm=norm, num_groups=num_groups, use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
            
            if i != 0:
                self.ups.append(Upsample(now_channels))
        
        assert len(channels) == 0
        
        self.out_norm = get_norm(norm, base_channels, num_groups)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)
    
    def forward(self, x, time=None, y=None):
        # Whether to apply padding to the input
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 4)

        # Fully connected layer for time input
        if self.time_mlp is not None:
            if time is None:
                raise ValueError("time conditioning was specified but time is not passed")
            time_emb = self.time_mlp(time)
        else:
            time_emb = None
        
        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")
        
        # First convolution for input image
        x = self.init_conv(x)

        # skips is used to store intermediate layers during downsampling
        skips = [x]
        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)
        
        # Feature integration and extraction
        for layer in self.mid:
            x = layer(x, time_emb, y)
        
        # Upsampling and feature fusion
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                skip = skips.pop()
                # During the upsampling phase, check and adjust tensor dimensions
                if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
                    diff_h = skip.shape[2] - x.shape[2] # Calculate the difference in height
                    diff_w = skip.shape[3] - x.shape[3] # Calculate the difference in width
                    x = F.pad(x, (0, diff_w, 0, diff_h))
                x = torch.cat([x, skip], dim=1)  
            x = layer(x, time_emb, y)

        # Upsampling and feature fusion
        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)
        
        if self.initial_pad != 0:
            return x[:, :, ip:-ip, ip:-ip]
        else:
            return x

