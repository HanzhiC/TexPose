# code modified based on GRAF: https://github.com/autonomousvision/graf
import torch
import torch.nn as nn
import random
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, opt, ndf=64):
        super(Discriminator, self).__init__()
        nc = 3
        self.imsize = opt.patch_size
        self.scale_conditional = opt.gan.scale_conditional
        self.geo_conditional = opt.gan.geo_conditional
        self.L_nocs = opt.gan.L_nocs
        self.L_normal = opt.gan.L_normal
        self.L_scale = opt.gan.L_scale
        self.c2f = opt.gan.geo_c2f is not None
        if self.c2f:
            assert self.geo_conditional and (self.L_nocs or self.L_normal)
            self.c2f_range = opt.gan.geo_c2f
        if self.geo_conditional: nc += 2 * 3
        if self.L_nocs: nc += self.L_nocs * 2 * 3
        if self.L_normal: nc += self.L_normal * 2 * 3

        self.progress = torch.nn.Parameter(torch.tensor(0.))
        assert (self.imsize == 16 or self.imsize == 32 or self.imsize == 64 or self.imsize == 128)

        SN = torch.nn.utils.spectral_norm
        IN = lambda x: nn.InstanceNorm2d(x)

        if self.scale_conditional:
            final_dim = ndf
            self.final = nn.Sequential(
                nn.LeakyReLU(0.2),
                SN(nn.Conv2d(ndf + self.L_scale * 2 + 1, ndf, (1, 1), (1, 1), (0, 0), bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                SN(nn.Conv2d(ndf, ndf, (1, 1), (1, 1), (0, 0), bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                SN(nn.Conv2d(ndf, 1, (1, 1), (1, 1), (0, 0), bias=False)),
            )
        else:
            final_dim = 1

        blocks = []
        if self.imsize == 128:
            blocks += [
                # input is (nc) x 128 x 128
                SN(nn.Conv2d(nc, ndf // 2, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # input is (ndf//2) x 64 x 64
                SN(nn.Conv2d(ndf // 2, ndf, (4, 4), (2, 2), (1, 1), bias=False)),
                IN(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                SN(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 4),
                IN(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize == 64:
            blocks += [
                # input is (nc) x 64 x 64
                SN(nn.Conv2d(nc, ndf, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                SN(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 4),
                IN(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize == 32:
            blocks += [
                # input is (nc) x 32 x 32
                SN(nn.Conv2d(nc, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                SN(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 4),
                IN(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize == 16:
            blocks += [
                # state size. (ndf*2) x 16 x 16
                SN(nn.Conv2d(nc, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 4),
                IN(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            raise NotImplementedError
        blocks += [
            # state size. (ndf*4) x 8 x 8
            SN(nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            # nn.BatchNorm2d(ndf * 8),
            IN(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SN(nn.Conv2d(ndf * 8, final_dim, (4, 4), (1, 1), (0, 0), bias=False)),
            # nn.Sigmoid()
        ]
        blocks = [x for x in blocks if x]
        self.main = nn.Sequential(*blocks)

    def forward(self, opt, x, scale=None):

        # process the raw input
        if self.geo_conditional:
            image, nocs, normal = x.split(3, dim=1)
            assert image.shape == nocs.shape == normal.shape
        else:
            image = x

        # generate the final input
        if self.geo_conditional:
            inputs = torch.cat([image, nocs, normal], dim=1)
            if self.L_nocs is not None:
                nocs_encode = self.positional_encoding(opt, nocs, L=self.L_nocs, reshape=True, c2f=self.c2f)
                inputs = torch.cat([inputs, nocs_encode], dim=1)
            if self.L_normal is not None:
                normal_encode = self.positional_encoding(opt, normal, L=self.L_nocs, reshape=True, c2f=self.c2f)
                inputs = torch.cat([inputs, normal_encode], dim=1)
        else:
            inputs = image
        out = self.main(inputs)  # [N, c1, 1, 1]

        if self.scale_conditional:
            scale_encode = self.positional_encoding(opt, scale, L=self.L_scale, reshape=True)
            out = torch.cat((out, scale_encode, scale), 1)  # [N, c1+c2, 1, 1]
            out = self.final(out).flatten()
        return out

    def positional_encoding(self, opt, x, L, reshape=False, c2f=False):  # [B,...,N]
        if reshape:
            batch_size, C_init, h, w = x.shape
            x = x.view(batch_size, C_init, h * w).permute(0, 2, 1)  # [B, hw, N]
        else:
            batch_size, _, C_init = x.shape
        encode_shape = x.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=opt.device) * np.pi  # [L]
        spectrum = x[..., None] * freq  # [B,hw,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,hw,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,hw,N,2,L]
        input_enc = input_enc.view(*encode_shape[:-1], -1)  # [B,hw,2NL]
        if c2f:
            # set weights for different frequency bands
            start, end = self.c2f_range
            alpha = (self.progress.data - start) / (end - start) * L
            k = torch.arange(L, dtype=torch.float32, device=opt.device)
            weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1, L) * weight).view(*shape)

        if reshape:
            input_enc = input_enc.permute(0, 2, 1).view(batch_size, -1, h, w)
            assert input_enc.shape[1] == 2 * C_init * L
        return input_enc

    def __call__(self, opt, x, scale=None):
        return self.forward(opt, x, scale)
