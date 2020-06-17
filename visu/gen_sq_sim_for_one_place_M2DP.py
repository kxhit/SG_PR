# from utils import tab_printer
# from sg_net import SGTrainer
# from parser_sg import parameter_parser
# import pykitti
import numpy as np
from tqdm import tqdm
import os
import json
from matplotlib import pyplot as plt
from sklearn import metrics
import torch
import time
from eval.compare.eval_tool import *


def main():
    list_dir = "visu"

    sequences = ["08"]  # test set


    for sequence in tqdm(sequences):
        print("sequence: ", sequence)
        # todo
        p_thresh = "5_20"
        output_path = list_dir + "/visu_sim/" + p_thresh + "/" + sequence + "/"
     
        print(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        fix_id = "000714"

        # choose feature database
        M2DP_path = "eval/compare/M2DP/feature_database/drop0/" + sequence + "_M2DP.bin"
        M2DP_data = np.fromfile(M2DP_path)
        M2DP_data = np.array(M2DP_data.reshape(192, -1), dtype=float)
        frame_nums = M2DP_data.shape[1]

        gt_db = []
        pred_db = []

        for i in range(frame_nums):
            f_distance = f_dist(M2DP_data[:, int(fix_id)], M2DP_data[:, int(i)], type="l2")
            pred = 1.0 / (f_distance + 1e-12)  # TODO: normalize distance?
            pred_db.append(pred)

        pred_db_path = output_path + sequence + "_M2DP_db.npy"
        # np.save(gt_db_path, gt_db)
        np.save(pred_db_path, pred_db)


if __name__ == "__main__":
    main()