from utils import tab_printer
# from simgnn import SimGNNTrainer #TODO
# from parser import parameter_parser #TODO
# from simgnn_dgcnn_batch import SimGNNTrainer
from sg_net import SGTrainer
from parser_sg import parameter_parser
# import pykitti
import numpy as np
from tqdm import tqdm
import os
import json
from matplotlib import pyplot as plt
from sklearn import metrics
import torch
import time


def main():
    list_dir = "visu"
    # todo
    graph_dir = "/media/datasets/kx/graph/"

    sequences = ["08"]  # test set

    batch_size = 128 # batch-size 128
    #init simgnn model
    args = parameter_parser()
    tab_printer(args)
    # todo
    # ds
    # model = "/media/datasets/kx/code/SimGNN_semantic/log_fold00_0030_xyz_sem_ds/116_best.pth"
    # model = "/media/datasets/kx/code/SimGNN_semantic/log_fold02_0030_xyz_sem_ds/106_best.pth"
    # model = "/media/datasets/kx/code/SimGNN_semantic/log_fold05_0030_xyz_sem_ds/110_best.pth"
    # model = "/media/datasets/kx/code/SimGNN_semantic/log_fold06_0030_xyz_sem_ds/12_best.pth"
    # model = "/media/datasets/kx/code/SimGNN_semantic/log_fold07_0030_xyz_sem_ds/46_best.pth"
    # model = "/media/datasets/kx/code/SimGNN_semantic/log_fold08_0030_xyz_sem_ds/30_best.pth"
    model = ""

    trainer = SGTrainer(args, model)
    trainer.model.eval()

    for sequence in tqdm(sequences):
        print("sequence: ", sequence)
        # todo
        output_path = list_dir + "/visu_sim/" + sequence + "/"
        # output_path = list_dir + "/xyz_sem/list/rangenet_ds/drop0_rotate/" + sequence + "/"
        # output_path = list_dir + "/xyz_sem/list/rangenet_ds/drop0/" + sequence + "/"
        # output_path = list_dir + "/xyz_sem/list/rangenet_ds/drop30/" + sequence + "/"
        # output_path = list_dir + "/xyz_sem/list/rangenet_ds/drop90/" + sequence + "/"

        # output_path = list_dir + "/xyz_sem/list/ds/drop0_rotate/" + sequence + "/"
        # output_path = list_dir + "/xyz_sem/list/ds/drop0/" + sequence + "/"
        # output_path = list_dir + "/xyz_sem/list/ds/drop30/" + sequence + "/"
        # output_path = list_dir + "/xyz_sem/list/ds/drop90/" + sequence + "/"

        # output_path = list_dir + "/xyz_sem/list/his/drop0/" + sequence + "/"

        # output_path = list_dir + "/simgnn_dgcnn_batch/list/drop0_rotate/" + sequence + "/"
        print(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        sq_dir = os.path.join(graph_dir, sequence)
        sq_graphs = os.listdir(sq_dir)
        sq_graphs.sort()
        fix_id = "000714.json"
        fix_graph_path = os.path.join(sq_dir, fix_id)
        fix_graph = json.load(open(fix_graph_path))
        # fix_pose = pykitti.odometry()

        gt_db = []
        pred_db = []

        graph_pairs = []
        for graph in sq_graphs:
            graph_path = os.path.join(sq_dir, graph)
            graph_i = json.load(open(graph_path))
            pair = {}
            pair["nodes_1"] = fix_graph["nodes"]
            pair["nodes_2"] = graph_i["nodes"]
            pair["centers_1"] = fix_graph["centers"]
            pair["centers_2"] = graph_i["centers"]
            # dist = np.linalg.norm(pair["centers_1"]-pair["centers_2"])
            pair["distance"] = 1    # fake gt


            graph_pairs.append(pair)

        batches = [graph_pairs[graph:graph + batch_size] for graph in
                   range(0, len(graph_pairs), batch_size)]
        for batch in tqdm(batches):
            pred, gt = trainer.eval_batch_pair_data(batch)  # todo fix 20
            pred_db.extend(pred)
            # gt_db.extend(gt)

        pred_db_path = output_path + sequence + "_DL_db.npy"
        # np.save(gt_db_path, gt_db)
        np.save(pred_db_path, pred_db)


if __name__ == "__main__":
    main()