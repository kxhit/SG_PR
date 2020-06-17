import os
import numpy as np
import json
import pykitti
from sklearn import metrics
from eval.compare.gen_gt import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from eval.compare.eval_tool import *
from Distance_SC import distance_sc

if __name__ == '__main__':
    list_dir = "/media/datasets/kx/code/SimGNN_semantic/eval/5_20/"
    pair_dir = "/media/datasets/yxm/graph_pairs_sem_5_20_ds"
    # pair_dir = "/media/work/data/kitti/odometry/semantic-kitti/DGCNN_graph_pairs_3_20_drop90_closure"
    # list_dir = "/media/work/3D/SimGNN/kx/SimGNN_semantic/pair_list_sample/3_20/"
    # pair_dir = "/media/work/data/kitti/odometry/semantic-kitti/DGCNN_graph_pairs_3_20/train_all/"
    # sequences = ["00", "02", "05", "08"]
    # sequences = ["00", "02", "05", "08", "06", "07"]
    # #choose sequence
    sequences = ["00"]
    for sequence in tqdm(sequences):
        print("sequence: ", sequence)
        # choose output path

        output_path = "/media/datasets/kx/code/SimGNN_semantic/eval/compare/ScanContext/list/5_20/drop0/" + sequence + "/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        list_path = list_dir + "draw_curve_" + sequence + ".npy"
        pairs_files = np.load(list_path).tolist()
        pairs_path = os.path.join(pair_dir, sequence)
        # choose feature database
        SC_path = "/media/datasets/kx/code/SimGNN_semantic/eval/compare/ScanContext/feature_database/drop0/" + sequence + "_SC.npy"
        SC_data = np.load(SC_path)
        frame_nums = SC_data.shape[0]
        print("frame_nums: ", frame_nums)

        #get time_stamp
        dataset = get_dataset(sequence_id=sequence)
        gt_db = []
        pred_db = []
        for pair in tqdm(pairs_files):
            data = json.load(open(os.path.join(pairs_path, pair)))
            if data["distance"] <= 5:  #TODO
                gt = 1
            elif data["distance"] >= 20:
                gt = 0
            else:
                print("wrong pair: ", pair)
                exit(-1)

            [i, j] = pair.split('.')[0].split('_')
            f_distance = distance_sc(SC_data[int(i), :, :], SC_data[int(j), :, :])
            pred = 1.0 / (f_distance + 1e-12)  # TODO: normalize distance?
            # gt = find_gt(i,j,gt_data_sq)
            pred_db.append(pred)
            gt_db.append(gt)

        assert len(pred_db) == len(gt_db)
        assert np.sum(gt_db) > 0 # gt_db should have positive samples
        pred_db = np.array(pred_db)
        gt_db = np.array(gt_db)
        #save results
        gt_db_path = output_path + sequence + "_gt_db.npy"
        pred_db_path = output_path + sequence + "_SC_db.npy"
        np.save(gt_db_path, gt_db)
        np.save(pred_db_path, pred_db)

        #####ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(gt_db, pred_db)
        roc_auc = metrics.auc(fpr, tpr)
        print("fpr: ", fpr)
        print("tpr: ", tpr)
        print("thresholds: ", roc_thresholds)
        print("roc_auc: ", roc_auc)

        # plot ROC Curve
        plt.figure(0)
        lw = 2
        # plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('SC ROC Curve')
        plt.legend(loc="lower right")
        roc_out = output_path + sequence + "_SC_roc_curve.png"
        plt.savefig(roc_out)
        plt.show()


        #### P-R
        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt_db, pred_db)
        # plot p-r curve
        plt.figure(1)
        lw = 2
        # plt.figure(figsize=(10, 10))
        plt.plot(recall, precision, color='darkorange',
                 lw=lw, label='P-R curve')  ###假正率为横坐标，真正率为纵坐标做曲线
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('SC Precision-Recall Curve')
        plt.legend(loc="lower right")
        pr_out = output_path + sequence + "_SC_pr_curve.png"
        plt.savefig(pr_out)
        plt.show()

