import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)
import numpy as np
import soundfile as sf
import pickle
import os
from time import time

if __name__ == "__main__":
    position_num_x = 13
    position_num_y = 13
    position_z = 1.35
    blank_space = 0.5
    mic_radius = 0.1
    mic_num = 4
    points_path = "./points.txt"
    minmax_path = "./minmax.pkl"
    results_dir = "./simulation_results/"

    # 残響時間と部屋の寸法
    rt60 = 0.5  # seconds
    room_dim = [7.0, 6.4, 2.7]  # meters ここを二次元にすると二次平面の部屋になります
    sampling_rate = 48000

    all_compute_time = 0
    all_write_time = 0
    source_num = position_num_x*position_num_y
    for source_index in range(source_num):
        # Sabineの残響式から壁面の平均吸音率と鏡像法での反射回数の上限を決めます
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

        # 部屋をつくります
        # fsは生成されるインパルス応答のサンプリング周波数です。入力する音源があるならそれに合わせる。
        room = pra.ShoeBox(
            room_dim, fs=sampling_rate, materials=pra.Material(e_absorption), max_order=max_order
        )

        interval_x = (room_dim[0] - blank_space*2)/(position_num_x - 1)
        interval_y = (room_dim[1] - blank_space*2)/(position_num_y - 1)
        assert mic_radius < interval_x
        assert mic_radius < interval_y
        x_array = np.arange(position_num_x)*(interval_x) + blank_space
        y_array = np.arange(position_num_y)*(interval_y) + blank_space

        positions = list()
        for x in x_array:
            for y in y_array:
                positions.append([x, y, position_z])
        positions = np.asarray(positions)

        if source_index == 0:
            points_num = 0
            points = ""
            for pos in positions:
                points += str(points_num) + "\t" + str(pos[0]) + "\t" + str(pos[1]) + "\t" + str(pos[2]) + "\n"
                points_num += 1
            
            with open(points_path, mode="wt", encoding="utf-8") as f:
                f.write(points)
            
            min_xyz = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
            max_xyz = np.asarray(room_dim, dtype=np.float32)
            minmax = (min_xyz, max_xyz)

            with open(minmax_path, mode="wb") as f:
                pickle.dump(minmax, f)

        ## create directivity object
        #dir_num = mic_num
        #dir_objs = list()
        #for i in range(dir_num):
        #    dir_obj = CardioidFamily(
        #    orientation=DirectionVector(azimuth=90*i, colatitude=90, degrees=True),
        #    pattern_enum=DirectivityPattern.CARDIOID
        #    )
        #    dir_objs.append(dir_obj)

        mic_positions = np.zeros((3, 0))

        for i in range(len(positions)):
            if i == source_index:
                source_position = positions[i]
            else:
                position_circle_xy = pra.beamforming.circular_2D_array(center=positions[i][:2], M=mic_num, phi0=0, radius=mic_radius)
                z = np.ones((1, mic_num))*positions[i][2]
                position_circle = np.concatenate((position_circle_xy, z), axis=0)
                mic_positions = np.concatenate((mic_positions, position_circle), axis=1)

        # room にマイクを追加します
        room.add_microphone_array(mic_positions)

        # 音源ごとに座標情報を与え、`room`に追加していきます。
        room.add_source(source_position)

        before_compute = time()
        room.compute_rir()
        after_compute = time()

        before_write = time()
        skip_flag = False
        for i in range(len(positions)):
            if i == source_index:
                skip_flag = True
                continue
            else:
                for j in range(mic_num):
                    path = os.path.join(results_dir, str(source_index) + "_" + str(i) + "_" + str(j+1) + ".wav")
                    if skip_flag:
                        sf.write(file=path, data=room.rir[(i-1)*4+j][0], samplerate=sampling_rate)
                    else:
                        sf.write(file=path, data=room.rir[i*4+j][0], samplerate=sampling_rate)
        after_write = time()

        compute_time = after_compute - before_compute
        write_time = after_write - before_write
        print(
            "compute time: {:.2f}s".format(compute_time), 
            "write time: {:.2f}s".format(write_time)
            )
        all_compute_time += compute_time
        all_write_time += write_time
        print("source_index{}({:.2f}) done!".format(source_index, (source_index+1)/source_num))
        print(
            "compute time: {:.2f} minutes now".format(all_compute_time/60), 
            "write time: {:.2f} minutes now".format(all_write_time/60)
            )
    
    print(
        "compute time: {:.2f} minutes".format(all_compute_time/60), 
        "write time: {:.2f} minutes".format(all_write_time/60)
        )
    print("all done!")
        