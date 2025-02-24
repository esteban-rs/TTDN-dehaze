import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model import Gradient, TTSR
from model.Modules import conv1x1, conv3x3, ResBlock, SFE, MergeTail
    

class SR(nn.Module):
    def __init__(self, num_res_blocks, num_grad_blocks, n_feats, res_scale, top_k):
        super(SR, self).__init__()
        self.num_res_blocks   = num_res_blocks 
        self.num_res_blocks_g = num_grad_blocks
        self.n_feats          = n_feats
        self.SFE              = SFE(self.num_res_blocks[0], n_feats, res_scale)
        self.SFE_GRAD         = SFE(self.num_res_blocks_g[0], n_feats, res_scale)
        self.gradient         = Gradient.gradient()
        self.top_k            = top_k

        self.conv11_head = nn.ModuleList()
        for i in range(top_k):
            self.conv11_head.append(conv3x3(256 + n_feats, n_feats))

        self.RB11        = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels  = n_feats, 
                                      out_channels = n_feats,
                                      res_scale    = res_scale))
        self.conv11_tail = conv3x3(n_feats, n_feats)

        ### subpixel 1 -> 2
        self.conv12      = conv3x3(n_feats, n_feats*4)
        self.ps12        = nn.PixelShuffle(2)
        self.conv22_head = nn.ModuleList()
        for i in range(top_k):
            self.conv22_head.append(conv3x3(128 + n_feats, n_feats))
            
        self.ex12        = TTSR.CSFI2(n_feats)
        self.RB21        = nn.ModuleList()
        self.RB22        = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB21.append(ResBlock(in_channels  = n_feats, 
                                      out_channels = n_feats,
                                      res_scale    = res_scale))
            self.RB22.append(ResBlock(in_channels  = n_feats, 
                                      out_channels = n_feats,
                                      res_scale    = res_scale))

        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)
        self.conv23      = conv3x3(n_feats, n_feats*4)
        self.ps23        = nn.PixelShuffle(2)

        self.conv33_head = nn.ModuleList()
        for i in range(top_k):
            self.conv33_head.append(conv3x3(64 + n_feats, n_feats))

        self.ex123 = TTSR.CSFI3(n_feats)

        self.RB31 = nn.ModuleList()
        self.RB32 = nn.ModuleList()
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB31.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB32.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))

        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)
        self.merge_tail  = MergeTail(n_feats)
        
        # Gradd  
        self.conv12_grad = conv3x3(2 * n_feats, n_feats)
        self.grad_12     = nn.ModuleList()
        for i in range(self.num_res_blocks_g[1]):
            self.grad_12.append(ResBlock(in_channels  = n_feats, 
                                      out_channels = n_feats))
                                
        self.conv23_grad = conv3x3(2 * n_feats, n_feats)
        self.grad_23     = nn.ModuleList()
        for i in range(self.num_res_blocks_g[2]):
            self.grad_23.append(ResBlock(in_channels  = n_feats, 
                                      out_channels = n_feats))
        
        self.conv33_grad = conv3x3(2 * n_feats, n_feats)
        self.grad_33     = nn.ModuleList()
        for i in range(self.num_res_blocks_g[3]):
            self.grad_33.append(ResBlock(in_channels  = n_feats, 
                                      out_channels = n_feats))

        self.fuse        = conv3x3(2 * n_feats, n_feats)
                                
        self.fuse_tail1 = conv3x3(n_feats, n_feats//2)
        self.fuse_tail2 = conv1x1(n_feats//2, 3)
        

    def forward(self, x, S = None, T_lv3 = None, T_lv2 = None, T_lv1 = None):
        ### shallow feature extraction
        g = self.gradient((x + 1) /2)
        x = self.SFE(x)

        ### stage11
        x11 = x
        
        ### soft-attention
        f_lv1 = x11
        for i in range(len(T_lv3)) :
            scale_factor = f_lv1.shape[2] // S[i].shape[2]
            S1           = F.interpolate(S[i], scale_factor = scale_factor, mode='bicubic')
            x11_res      = torch.cat((f_lv1, T_lv3[i] * S1), dim = 1)
            x11_res      = self.conv11_head[i](x11_res) * S1
            x11          = x11 + x11_res
        x11_res = x11  
        
        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res
        
        ### stage21, 22
        x21     = x11
        x21_res = x21
        x22     = self.conv12(x11)
        x22     = F.relu(self.ps12(x22))
        
        ### soft-attention
        f_lv2 = x22
        for i in range(len(T_lv3)) :
            scale_factor = f_lv2.shape[2] // S[i].shape[2]
            S2           = F.interpolate(S[i], scale_factor = scale_factor, mode='bicubic')
            x22_res      = torch.cat((f_lv2, T_lv2[i] * S2), dim=1)
            x22_res      = self.conv22_head[i](x22_res) * S2
            x22          = x22 + x22_res        
        x22_res = x22

        x21_res, x22_res = self.ex12(x21_res, x22_res)
        

        for i in range(self.num_res_blocks[2]):
            x21_res = self.RB21[i](x21_res)
            x22_res = self.RB22[i](x22_res)

        x21_res = self.conv21_tail(x21_res)
        x22_res = self.conv22_tail(x22_res)
        x21     = x21 + x21_res
        x22     = x22 + x22_res
        
        ### stage31, 32, 33
        x31     = x21
        x31_res = x31
        x32     = x22
        x32_res = x32
        x33     = self.conv23(x22)
        x33     = F.relu(self.ps23(x33))
        
        f_lv3 = x33
        for i in range(len(T_lv3)) :
            scale_factor = f_lv3.shape[2] // S[i].shape[2]
            S3           = F.interpolate(S[i], scale_factor=scale_factor, mode='bicubic')
            x33_res      = torch.cat((f_lv3, T_lv1[i] * S3), dim=1)
            x33_res      = self.conv33_head[i](x33_res) * S3 
            x33          = x33 + x33_res
        x33_res = x33

        x31_res, x32_res, x33_res = self.ex123(x31_res, x32_res, x33_res)

        for i in range(self.num_res_blocks[3]):
            x31_res = self.RB31[i](x31_res)
            x32_res = self.RB32[i](x32_res)
            x33_res = self.RB33[i](x33_res)

        x31_res = self.conv31_tail(x31_res)
        x32_res = self.conv32_tail(x32_res)
        x33_res = self.conv33_tail(x33_res)
        
        x31 = x31 + x31_res
        x32 = x32 + x32_res
        x33 = x33 + x33_res

        T1 = x31
        T2 = x32
        T3 = x33
        
        x_tt = self.merge_tail(T1, T2, T3)   
                
        ## grad
        x_grad = self.SFE_GRAD(g)
                
        # fuse level 1
        x_grad1 = torch.cat([x_grad, T1], dim = 1)
        x_grad1 = F.relu(self.conv12_grad(x_grad1))
        
        x_grad1_res = x_grad1
        for i in range(self.num_res_blocks_g[1]):
            x_grad1_res = self.grad_12[i](x_grad1_res)
        x_grad1 = x_grad1 + x_grad1_res

        # fuse level 2
        x_grad1 = F.interpolate(x_grad1, scale_factor = 2, mode='bicubic')
        x_grad2 = torch.cat([x_grad1, T2], dim = 1)
        x_grad2 = F.relu(self.conv23_grad(x_grad2))

        x_grad2_res = x_grad2
        for i in range(self.num_res_blocks_g[2]):
            x_grad2_res = self.grad_23[i](x_grad2_res)
        x_grad2 = x_grad2 + x_grad2_res

        # fuse level 3
        x_grad2 = F.interpolate(x_grad2, scale_factor = 2, mode='bicubic')
        x_grad3 = torch.cat([x_grad2, T3], dim = 1)
        x_grad3 = F.relu(self.conv33_grad(x_grad3))
        
        x_grad3_res = x_grad3
        for i in range(self.num_res_blocks_g[3]):
            x_grad3_res = self.grad_33[i](x_grad3_res)
        x_grad3 = x_grad3 + x_grad3_res

        x_cat = torch.cat([x_tt, x_grad3], dim = 1)
        x_cat = F.relu(self.fuse(x_cat))
        
        x_cat = self.fuse_tail1(x_cat)
        x_cat = self.fuse_tail2(x_cat)
        
        return torch.clamp(x_cat, -1, 1)