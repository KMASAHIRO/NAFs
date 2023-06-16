import torch
from torch import nn
import math
import numpy as np
from model.modules import fit_predict_torch

class basic_project2(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(basic_project2, self).__init__()
        self.proj = nn.Linear(input_ch, output_ch, bias=True)
    def forward(self, x):
        return self.proj(x)

class kernel_linear_act(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(kernel_linear_act, self).__init__()
        self.block = nn.Sequential(nn.LeakyReLU(negative_slope=0.1), basic_project2(input_ch, output_ch))
    def forward(self, input_x):
        return self.block(input_x)

class kernel_residual_fc_embeds(nn.Module):
    def __init__(self, input_ch, intermediate_ch=512, grid_ch = 64, num_block=8, output_ch=1, grid_gap=0.25, grid_bandwidth=0.25, bandwidth_min=0.1, bandwidth_max=0.5, float_amt=0.1, min_xyz=None, max_xyz=None, probe=False):
        super(kernel_residual_fc_embeds, self).__init__()
        # input_ch (int): number of ch going into the network
        # intermediate_ch (int): number of intermediate neurons
        # min_xyz, max_xyz are the bounding box of the room in real (not normalized) coordinates
        # probe = True returns the features of the last layer

        self.proj = basic_project2(input_ch + int(2*grid_ch), intermediate_ch)
        self.residual_1 = nn.Sequential(basic_project2(input_ch + 128, intermediate_ch), nn.LeakyReLU(negative_slope=0.1), basic_project2(intermediate_ch, intermediate_ch))
        self.layers = torch.nn.ModuleList()
        for k in range(num_block - 2):
            self.layers.append(kernel_linear_act(intermediate_ch, intermediate_ch))

        self.out_layer = nn.Linear(intermediate_ch, output_ch)
        self.blocks = len(self.layers)
        self.probe = probe

        ### Make the grid

        grid_coors_x = np.arange(min_xyz[0], max_xyz[0], grid_gap)
        grid_coors_y = np.arange(min_xyz[1], max_xyz[1], grid_gap)
        grid_coors_z = np.arange(min_xyz[2], max_xyz[2], grid_gap)
        grid_coors_x, grid_coors_y, grid_coors_z = np.meshgrid(grid_coors_x, grid_coors_y, grid_coors_z)
        grid_coors_x = grid_coors_x.flatten()
        grid_coors_y = grid_coors_y.flatten()
        grid_coors_z = grid_coors_z.flatten()
        xyz_train = np.array([grid_coors_x, grid_coors_y, grid_coors_z]).T
        self.bandwidth_min = bandwidth_min
        self.bandwidth_max = bandwidth_max
        self.float_amt = float_amt
        self.bandwidths = nn.Parameter(torch.zeros(len(grid_coors_x))+grid_bandwidth, requires_grad=True)
        self.register_buffer("grid_coors_xyz",torch.from_numpy(xyz_train).float(), persistent=True)
        self.xyz_offset = nn.Parameter(torch.zeros_like(self.grid_coors_xyz), requires_grad=True)
        self.grid_0 = nn.Parameter(torch.randn(len(grid_coors_x),grid_ch, device="cpu").float() / np.sqrt(float(grid_ch)), requires_grad=True)

    def forward(self, input_stuff, sound_loc=None):
        SAMPLES = input_stuff.shape[1]
        sound_loc_v0 = sound_loc[..., :3]
        sound_loc_v1 = sound_loc[..., 3:]

        # Prevent numerical issues
        self.bandwidths.data = torch.clamp(self.bandwidths.data, self.bandwidth_min, self.bandwidth_max)

        grid_coors_baseline = self.grid_coors_xyz + torch.tanh(self.xyz_offset) * self.float_amt
        grid_feat_v0 = fit_predict_torch(grid_coors_baseline, self.grid_0, sound_loc_v0, self.bandwidths)
        grid_feat_v1 = fit_predict_torch(grid_coors_baseline, self.grid_0, sound_loc_v1, self.bandwidths)
        total_grid = torch.cat((grid_feat_v0, grid_feat_v1), dim=-1).unsqueeze(1).expand(-1, SAMPLES, -1)

        my_input = torch.cat((total_grid, input_stuff), dim=-1)
        out = self.proj(my_input).unsqueeze(2).repeat(1, 1, 1, 1)
        for k in range(len(self.layers)):
            out = self.layers[k](out)
            if k == (self.blocks // 2 - 1):
                out = out + self.residual_1(my_input).unsqueeze(2).repeat(1, 1, 1, 1)
        if self.probe:
            return out
        return self.out_layer(out)