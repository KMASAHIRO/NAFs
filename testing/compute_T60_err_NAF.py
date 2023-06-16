from inspect import getsourcefile
import os
import numpy as np
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from testing.test_utils import get_wave, compute_t60, load_audio, get_wave_2
from options import Options
import h5py
import pickle
for apt in ['apartment_1', 'apartment_2', 'frl_apartment_2', 'frl_apartment_4', 'office_4', 'room_2']:
    output_wav = dict()
    cur_args = Options().parse()
    cur_args.apt = apt

    exp_name = cur_args.exp_name

    apt = cur_args.apt

    exp_name_filled = exp_name.format(cur_args.apt)
    exp_name_filled = exp_name.format(cur_args.apt)
    cur_args.exp_name = exp_name_filled

    exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
    cur_args.exp_dir = exp_dir

    result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
    cur_args.result_output_dir = result_output_dir

    save_name = os.path.join(cur_args.result_output_dir, cur_args.apt+"_NAF.pkl")# +"dual_grid1"
    saver_obj = h5py.File(save_name, "r")

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
    all_t60 = []
    offset = 0
    for k in all_keys:
        offset += 1
        if offset%1000==0:
            print(offset)
        net_out = saver_obj[k + "_out_mag"][:]
        gt_out = saver_obj[k + "_gt_mag"][:]
        actual_spec_len = net_out.shape[-1]
        std_ = std[:, :, :actual_spec_len]
        mean_ = mean[:, :, :actual_spec_len]

        net_out = (net_out * std_ + mean_)[0]
        gt_out = (gt_out * std_ + mean_)[0]

        net_phase = saver_obj[k + "_out_phase"][:][0]*3.0*0.58
        gt_phase = saver_obj[k + "_gt_phase"][:][0]*3.0*0.58
        orientation = str([0, 90, 180, 270][int(k.split("_")[0])])
        node_names = k.split("_")[1].replace("[", "").replace("]", "").replace("'", "").split(",")
        first = str(int(node_names[0]))
        second = str(int(node_names[1]))
        audio_file_name = os.path.join(cur_args.wav_base, apt, orientation, "{}_{}.wav".format(first, second))
        gt_wav2 = load_audio(audio_file_name)
        ##################
        net_wav = get_wave_2(net_out, net_phase)
        gt_wav = get_wave_2(gt_out, gt_phase)
        t60s = compute_t60(gt_wav, net_wav)
        all_t60.append(t60s)
        output_wav[k] = {"net_wav": net_wav, "gt_wav": gt_wav}

    t60s_np = np.array(all_t60)

    diff = np.abs(t60s_np[:,2:]-t60s_np[:,:2])/np.abs(t60s_np[:,:2])
    # diff = np.abs((t60s_np[:,2:]-t60s_np[:,:2])/t60s_np[:,:2])
    mask = np.any(t60s_np<-0.5, axis=1)
    diff = np.mean(diff, axis=1)
    diff[mask] = 1

    with open("./results/inference_wav/" + apt + "_NAF.pkl", mode="wb") as f:
        pickle.dump(output_wav, f)
    print("{}:{} total invalids out of {}".format(apt, np.sum(mask), mask.shape[0]))
    print(np.mean(diff)*100)