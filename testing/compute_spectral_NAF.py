import pickle
from inspect import getsourcefile
import os

import os.path as path, sys

import numpy as np

current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from testing.test_utils import spectral
from options import Options
import h5py


for apt in ['apartment_1', 'apartment_2', 'frl_apartment_2', 'frl_apartment_4', 'office_4', 'room_2']:
    cur_args = Options().parse()
    cur_args.apt = apt
    exp_name = cur_args.exp_name
    exp_name_filled = exp_name.format(cur_args.apt)
    exp_name_filled = exp_name.format(cur_args.apt)
    cur_args.exp_name = exp_name_filled

    exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
    cur_args.exp_dir = exp_dir

    result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
    cur_args.result_output_dir = result_output_dir

    save_name = os.path.join(cur_args.result_output_dir, cur_args.apt+"_NAF.pkl")
    saver_obj = h5py.File(save_name, "r")
    # with open(data_path, "rb") as f:
    #     data = pickle.load(f)

    std = saver_obj["std"][:]+0.0
    mean = saver_obj["mean"][:]+0.0

    keys = list(saver_obj.keys())
    keys_new = []
    for k in keys:
        if not k in ["mean", "std"]:
            keys_new.append(k.split("]")[0]+"]")
    all_keys = list(set(keys_new))

    loss = 0
    total = 0
    phase_loss = 0
    def get_stats(in_val):
        return np.mean(in_val), np.max(in_val), np.min(in_val)

    offset = 0
    for k in all_keys:
        offset += 1
        if offset %1000==0:
            print(offset)
        net_out = saver_obj[k+"_out_mag"][:]
        gt_out = saver_obj[k+"_gt_mag"][:]
        actual_spec_len = net_out.shape[-1]
        std_ = std[:, :, :actual_spec_len]
        mean_ = mean[:, :, :actual_spec_len]

        net_out_phase = saver_obj[k+"_out_phase"][:]*3.0*0.58
        gt_out_phase = saver_obj[k+"_gt_phase"][:]*3.0*0.58
        phase_loss += spectral(net_out_phase, gt_out_phase)

        net_out = (net_out*std_ + mean_)[0]
        gt_out = (gt_out * std_ + mean_)[0]
        loss += spectral(net_out, gt_out)
        total += 1.0

    mean_loss = loss/total
    mean_loss_phase = phase_loss/total
    print("{} the spectral loss is {}".format(cur_args.apt, mean_loss))
    print("{} the spectral phase loss is {}".format(cur_args.apt, mean_loss_phase))