import numpy as np
import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            '{}'.format(self.__class__.__name__) + '({},'.format(self.weight.shape[1]) + '{},'.format(self.weight.shape[0]) + 
'{},'.format(self.weight.shape[2]) + 'stride={},'.format(self.stride) + ' padding={})'.format(self.padding)
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            self.__class__.__name__ + '({},'.format(self.weight.shape[1]) +  '{})'.format(self.weight.shape[0])
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class EmbProb(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
    def forward(self, input):
        batch = input.shape[0]
        out = torch.ones(batch,self.in_channel, device=input.get_device())
        for i in range(batch):
            out[i,:np.int(torch.round(input[i]*8)/8.0*self.in_channel*0.5)] = 0
        return out

class InitModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        g = 16,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.g = g


        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.prob_emb = EmbProb(in_channel)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            '{}({}, {}, {}, upsample={}, downsample={})'.format(self.__class__.__name__,self.in_channel,self.out_channel,self.kernel_size,self.upsample,self.downsample)
        )

    def forward(self, input, prob, style):


        batch, in_channel, height, width = input.shape
        channels = self.out_channel
        groups = self.g

 

        style1 = style
        style2 = style
        if style.shape[1]>512:
            style1, style2 = torch.chunk(style,2,1)

        style = self.modulation(style2).view(batch, 1, in_channel)#*style_base

        style = style.view(batch, 1, in_channel, 1, 1)#*style_base
        if type(prob)==type(None):
            weight = self.scale * self.weight * style
        else:

            prob = self.prob_emb(prob)
            prob = prob.view(batch, 1, in_channel, 1, 1)

            weight = self.scale * self.weight * style * prob


        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)



        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        tmp = F.normalize(self.weight.reshape(channels,-1), dim=1).reshape((1,channels,-1))
        co = torch.bmm(tmp,torch.transpose(tmp, 1,2)).reshape((1,channels,channels))

        return out, co.detach()


class GroupModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        g = 16,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.g = g


        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation_bases = EqualLinear(style_dim, g, bias_init=1)
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.prob_emb = EmbProb(in_channel)

        self.demodulate = demodulate

        self.cluster = nn.Parameter(torch.randn(g, in_channel))

        self.gate = nn.Parameter(torch.ones(1)*-1)

        self.tmpr = nn.Parameter(torch.ones(1)*-2)




    def __repr__(self):
        return (
            '{}({}, {}, {}, upsample={}, downsample={})'.format(self.__class__.__name__,self.in_channel,self.out_channel,self.kernel_size,self.upsample,self.downsample)
        )

    def forward(self, input, co, prob, style):


        batch, in_channel, height, width = input.shape
        channels = self.out_channel
        groups = self.g

        cluster = self.cluster
        tmpr = torch.sigmoid(self.tmpr)*5
        gate = torch.sigmoid(self.gate)

        S = torch.sigmoid((co-gate)*50)
        cluster = F.softmax(tmpr*cluster,dim=0)
        loss1 = (torch.matmul(cluster.T,cluster)-S.detach()).pow(2).sum()
        loss2 = (torch.tensor(in_channel*in_channel/self.g, device = input.device) - S.sum()).pow(2).sum()

        grouped = torch.round(F.softmax(cluster*500,dim=0)).detach() + 1e-6
        style1, style2 = torch.chunk(style,2,1)

        style_base = self.modulation_bases(style1).view(batch, groups, 1)
        style = self.modulation(style2).view(batch, 1, in_channel)#*style_base
        gg = grouped.reshape(1,groups,in_channel).repeat(batch,1,1)
        tmp = F.normalize(gg*style,dim=2)
        style = tmp*style_base
        style = style.sum(1)
        style = style.view(batch, 1, in_channel, 1, 1)#*style_base
        if type(prob)==type(None):
            weight = self.scale * self.weight * style
        else:

            prob = self.prob_emb(prob)
            prob = prob.view(batch, 1, in_channel, 1, 1)

            weight = self.scale * self.weight * style * prob


        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)



        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        tmp = F.normalize(self.weight.reshape(channels,-1), dim=1).reshape((1,channels,-1))
        co = torch.bmm(tmp,torch.transpose(tmp, 1,2)).reshape((1,channels,channels))

        return out, co.detach(), loss1, loss2

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        g = 16,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.g = g


        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.prob_emb = EmbProb(in_channel)

        self.demodulate = demodulate



    def __repr__(self):
        return (
            '{}({}, {}, {}, upsample={}, downsample={})'.format(self.__class__.__name__,self.in_channel,self.out_channel,self.kernel_size,self.upsample,self.downsample)
        )

    def forward(self, input, co, prob, style):


        batch, in_channel, height, width = input.shape
        channels = self.out_channel
        groups = self.g

        style1, style2 = torch.chunk(style,2,1)

        style = self.modulation(style2).view(batch, 1, in_channel)#*style_base

        style = style.view(batch, 1, in_channel, 1, 1)#*style_base
        if type(prob)==type(None):
            weight = self.scale * self.weight * style
        else:

            prob = self.prob_emb(prob)
            prob = prob.view(batch, 1, in_channel, 1, 1)

            weight = self.scale * self.weight * style * prob


        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)



        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class InitStyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = InitModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, prob, style, noise=None):
        out, co = self.conv(input, prob, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out, co

class GroupStyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = GroupModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, co, prob, style, noise=None):
        out, co,loss1, loss2 = self.conv(input, co, prob, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out, co,loss1, loss2



class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, out, prob, style, skip=None):
        out = self.conv(input, out, prob, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=1,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp-1):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.final_style = EqualLinear(style_dim, style_dim*2, lr_mul=lr_mlp, activation='fused_lrelu')

        self.prob = EmbProb(512)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = InitStyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer('noise_{}'.format(layer_idx), torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                GroupStyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                GroupStyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        prob=None,
        return_loss=False,
        return_latents=False,
        return_info=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.final_style(self.style(s)) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, 'noise_{}'.format(i)) for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
            info_latent = styles[0]

        else:
            if random.randint(0,9)<50:
                if inject_index is None:
                    inject_index = random.randint(1, self.n_latent - 1)

                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

                latent = torch.cat([latent, latent2], 1)
                info_latent = None

            else:
                latent = styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)
                latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent, 1)

                latent = torch.cat([latent[:,:,:512], latent2[:,:,512:]], 2)
                info_latent = torch.cat([styles[0][:,:512], styles[1][:,512:]],1)


        out = self.input(latent)
        out, co = self.conv1(out, prob, latent[:, 0], noise=noise[0])
         
        skip = self.to_rgb1(out, co, prob, latent[:, 1])

        i = 1
        loss1 = None
        loss2 = None

        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out, co, l1, l2 = conv1(out, co, prob, latent[:, i], noise=noise1)
            if type(loss1)==type(None):
                loss1 = l1
                loss2 = l2
            else:
                loss1 = loss1 + l1
                loss2 = loss2 + l2
            out, co, l1, l2 = conv2(out, co, prob, latent[:, i + 1], noise=noise2)
            loss1 = loss1 + l1
            loss2 = loss2 + l2
            skip = to_rgb(out, co, prob, latent[:, i + 2], skip)
            i += 2
        loss1 = loss1/20
        loss2 = loss2/20
        image = skip

        if return_info and return_loss:
            return image, None, loss1, loss2, info_latent
        elif return_info:
            return image, None, None, None, info_latent
        elif return_latents:
            return image, latent, None, None, None
        else:
            return image, latent, loss1, loss2, info_latent

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)

        self.final_linear = EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu')
        self.final_info_linear = EqualLinear(channels[4], 1 + channels[4]*2)

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)
        out = self.final_info_linear(out)

        return out[:,0],out[:,1:]






