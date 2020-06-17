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
    p_thresh = 10
    sequences = ["01","02"]


    # #choose sequence
    # sequence = "00"
    for sequence in tqdm(sequences):
        print("sequence: ", sequence)
        list_dir = "/media/work/data/Ford/IJRR-Dataset-" + str(int(sequence)) + "/SG/" + str(p_thresh) + "_20/"
        pair_dir = "/media/work/data/Ford/IJRR-Dataset-" + str(int(sequence)) + "/SG/" + str(p_thresh) + "_20/graph_paris/"
        dataset_dir = "/media/work/data/Ford/IJRR-Dataset-" + str(int(sequence)) + "/SG"
        # choose output path
        output_path = "eval/Ford/" + str(p_thresh) + "_20/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        list_path = list_dir + "draw_curve.npy"
        pairs_files = np.load(list_path).tolist()
        # pairs_path = os.path.join(pair_dir, sequence)
        #choose feature database
        # PV_path = "/media/work/3D/PointNetVLAD/pointnetvlad/log_fold00/feature_database/drop0_fold00/"+sequence+"_PV_ref.npy"
        PV_path = "feature_database/"+sequence+"_PV.npy"
        PV_data = np.load(PV_path)
        frame_nums = PV_data.shape[0]
        print("frame_nums: ", frame_nums)
        #get time_stamp
        # dataset = get_dataset(sequence_id=sequence)
        gt_db = []
        pred_db = []
        for pair in tqdm(pairs_files):
            data = json.load(open(os.path.join(pair_dir, pair)))
            if data["distance"] <= p_thresh:
                gt = 1
            elif data["distance"] >= 20:
                gt = 0
            else:
                print("wrong pair: ", pair)
                exit(-1)

            [i,j] = pair.split('.')[0].split('_')
            if sequence == "02":
                f_distance = f_dist(PV_data[int(i)-1, :], PV_data[int(j)-1, :], type="l2")
            else:
                f_distance = f_dist(PV_data[int(i) - 75, :], PV_data[int(j) - 75, :], type="l2")
            pred = 1.0 / (f_distance + 1e-12)  # TODO: normalize distance?
            pred_db.append(pred)
            gt_db.append(gt)


        assert len(pred_db) == len(gt_db)
        assert np.sum(gt_db) > 0 # gt_db should have positive samples
        pred_db = np.array(pred_db)
        gt_db = np.array(gt_db)
        #save results
        gt_db_path = output_path + sequence + "_gt_db.npy"
        pred_db_path = output_path + sequence + "_PV_db.npy"
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
        plt.title('PV ROC Curve')
        plt.legend(loc="lower right")
        roc_out = output_path + sequence + "_PV_roc_curve.png"
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
        plt.title('PV Precision-Recall Curve')
        plt.legend(loc="lower right")
        pr_out = output_path + sequence + "_PV_pr_curve.png"
        plt.savefig(pr_out)
        plt.show()

        # calc F1-score
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        # print(F1_score)
        F1_max_score = np.max(F1_score)
        print("F1 max score: ", F1_max_score)
        f1_out = output_path + sequence + "_SC_F1_max.txt"
        with open(f1_out, "w") as out:
            out.write(str(F1_max_score))

