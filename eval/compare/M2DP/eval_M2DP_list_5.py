import os
import numpy as np
import json
# import pykitti
from sklearn import metrics
from eval.compare.gen_gt import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from eval.compare.eval_tool import *

if __name__ == '__main__':
    list_dir = "../../gt/5_20/"
    pair_dir = "/media/work/data/kitti/odometry/semantic-kitti/5_20/graph_pairs_sem_5_20_ds"
    # pair_dir = "/media/work/data/kitti/odometry/semantic-kitti/DGCNN_graph_pairs_3_20_drop90_closure"
    sequences = ["00", "02", "05", "08", "06", "07"]
    # #choose sequence
    # sequences = ["00"]
    for sequence in tqdm(sequences):
        print("sequence: ", sequence)
        # choose output path
        output_path = "eval/5_20/drop0/" + sequence + "/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        list_path = list_dir + "draw_curve_" + sequence + ".npy"
        pairs_files = np.load(list_path).tolist()
        pairs_path = os.path.join(pair_dir, sequence)
        #choose feature database
        M2DP_path = "feature_database/drop0/" + sequence + "_M2DP.bin"
        M2DP_data = np.fromfile(M2DP_path)
        M2DP_data = np.array(M2DP_data.reshape(192,-1),dtype=float)
        frame_nums = M2DP_data.shape[1]
        print("frame_nums: ", frame_nums)
        #get time_stamp
        # dataset = get_dataset(sequence_id=sequence)
        gt_db = []
        pred_db = []
        for pair in tqdm(pairs_files):
            data = json.load(open(os.path.join(pairs_path, pair)))
            if data["distance"] <= 5:
                gt = 1
            elif data["distance"] >= 20:
                gt = 0
            else:
                print("wrong pair: ", pair)
                exit(-1)

            [i,j] = pair.split('.')[0].split('_')
            f_distance = f_dist(M2DP_data[:, int(i)], M2DP_data[:, int(j)], type="l2")
            pred = 1.0 / (f_distance + 1e-12)  # TODO: normalize distance?
            pred_db.append(pred)
            gt_db.append(gt)


        assert len(pred_db) == len(gt_db)
        assert np.sum(gt_db) > 0 # gt_db should have positive samples
        pred_db = np.array(pred_db)
        gt_db = np.array(gt_db)
        #save results
        gt_db_path = output_path + sequence + "_gt_db.npy"
        pred_db_path = output_path + sequence + "_M2DP_db.npy"
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
        plt.title('M2DP ROC Curve')
        plt.legend(loc="lower right")
        roc_out = output_path + sequence + "_M2DP_roc_curve.png"
        plt.savefig(roc_out)
        # plt.show()
        plt.close()


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
        plt.title('M2DP Precision-Recall Curve')
        plt.legend(loc="lower right")
        pr_out = output_path + sequence + "_M2DP_pr_curve.png"
        plt.savefig(pr_out)
        # plt.show()
        plt.close()

