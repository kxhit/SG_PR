'''
Generate SCs from pointcloud, build and save sc_database to disk
'''

import numpy as np
from tqdm import tqdm
import os
from make_sc_example import *

if __name__ == "__main__":
    # sequences = ["00", "02", "05", "08"]
    sequences = ["00", "02", "05", "08", "06", "07"]
    # #choose sequence
    # sequence = "00"
    for sequence in tqdm(sequences):
        print("sequence: ", sequence)
        # choose output path
        # output_path = "/home/kx/project/3D/3D_loop_closure_detection/3D_dl_lc/eval/ScanContext/feature_database/fov100/"
        # output_path = "/home/kx/project/3D/3D_loop_closure_detection/3D_dl_lc/eval/ScanContext/feature_database/drop30/"
        output_path = "test_time"
        if not os.path.exists(output_path):
            os.makedirs(output_path)


        # bin_dir = '/media/data/kitti/odometry/dataset/sequences/' + sequence + "/velodyne/"
        bin_dir = '/media/work/data/kitti/odometry/semantic-kitti/semantic_kitti_label_100degrees/sequences/' + sequence + "/"
        bin_db = kitti_vlp_database(bin_dir)

        SC_db = []
        for bin_idx in tqdm(range(bin_db.num_bins)):
            bin_file_name = bin_db.bin_files[bin_idx]
            bin_path = bin_db.bin_dir + bin_file_name
            # sc = ScanContext(bin_dir, bin_file_name)
            # sc = ScanContext(bin_dir, bin_file_name, random_rotate=True) # calc SC
            sc = ScanContext(bin_dir, bin_file_name, random_rotate=False)  # calc SC
            # print(len(sc.SCs))
            # print(sc.SCs[0].shape)
            SC_db.append(sc.SCs[0])

        output_file = output_path + sequence + "_SC"
        np.save(output_file, SC_db)
        # print("sq: ", sequence)
        # print("SC: ", SC_db[:2])