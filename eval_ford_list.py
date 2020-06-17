from utils import tab_printer
from sg_net import SGTrainer
from parser_sg import parameter_parser
import numpy as np
from tqdm import tqdm
import os
import json
from matplotlib import pyplot as plt
from sklearn import metrics
import torch
import time


def main():
    batch_size = 128  # batch-size 128

    eval_dir = "eval/Ford"
    method = "RN"  # SK RN
    # ds
    # model = "/media/work/3D/SimGNN/kx/SG_LC/bk/3_20/log_fold07_0030_xyz_sem_ds/46_best.pth"
    # model = "/media/work/3D/SimGNN/kx/SG_LC/10_20/log_10_20_fold07/12_best.pth"
    # model = "/media/work/3D/SimGNN/kx/SG_LC/10_20/log_10_20_fold00/34_best.pth"
    # model = "/media/work/3D/SimGNN/kx/SG_LC/10_20/log_10_20_fold02/12_best.pth"
    # model = "/media/work/3D/SimGNN/kx/SG_LC/10_20/log_10_20_fold05/28_best.pth"
    # model = "/media/work/3D/SimGNN/kx/SG_LC/10_20/log_10_20_fold06/22_best.pth"
    model = "/media/work/3D/SimGNN/kx/SG_LC/10_20/log_10_20_fold07/12_best.pth"
    # model = "/media/work/3D/SimGNN/kx/SG_LC/10_20/log_10_20_fold08/20_best.pth"

    # init simgnn model
    args = parameter_parser()
    tab_printer(args)
    p_thresh = str(args.p_thresh)
    sequence = "02"
    # p_thresh = str(3)
    pair_dir = "/media/work/data/Ford/IJRR-Dataset-"+str(int(sequence))+"/SG/" + p_thresh + "_20/graph_paris/"
    list_path = "/media/work/data/Ford/IJRR-Dataset-"+str(int(sequence))+"/SG/" + p_thresh + "_20/draw_curve" + ".npy"
    trainer = SGTrainer(args, model)
    trainer.model.eval()

    # todo
    if method == "SK":
        output_path = eval_dir + "/" + p_thresh + "_20/" + "Ours_SK/" + "drop0/"
    elif method == "RN":
        output_path = eval_dir + "/" + p_thresh + "_20/" + "Ours_RN/" + "drop0/"
    else:
        print("method error: ", method)
        exit(-1)
    print(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pairs_files = np.load(list_path).tolist()


    gt_db = []
    pred_db = []

    graph_pairs = []
    for pair in tqdm(pairs_files):
        pair_file = os.path.join(pair_dir, pair)
        graph_pairs.append(pair_file)

    batches = [graph_pairs[graph:graph + batch_size] for graph in
               range(0, len(graph_pairs), batch_size)]
    for batch in tqdm(batches):
        pred, gt = trainer.eval_batch_pair(batch)
        pred_db.extend(pred)
        gt_db.extend(gt)

    assert len(pred_db) == len(gt_db)
    assert np.sum(gt_db) > 0  # gt_db should have positive samples


    # calc metrics

    pred_db = np.array(pred_db)
    gt_db = np.array(gt_db)
    # save results
    gt_db_path = output_path + sequence + "_gt_db.npy"
    pred_db_path = output_path + sequence + "_DL_db.npy"
    np.save(gt_db_path, gt_db)
    np.save(pred_db_path, pred_db)
    # id_db_path = output_path + sequence + "_id_db.npy"
    # np.save(id_db_path, id_db)
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
    plt.title('DL ROC Curve')
    plt.legend(loc="lower right")
    roc_out = output_path + sequence + "_DL_roc_curve.png"
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
    plt.axis([0,1,0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('DL Precision-Recall Curve')
    plt.legend(loc="lower right")
    pr_out = output_path + sequence + "_DL_pr_curve.png"
    plt.savefig(pr_out)
    # plt.show()
    plt.close()

    # calc F1-score
    F1_score = 2 * precision * recall / (precision + recall)
    F1_score = np.nan_to_num(F1_score)
    # print(F1_score)
    F1_max_score = np.max(F1_score)
    f1_out = output_path + sequence + "_DL_F1_max.txt"
    with open(f1_out, "w") as out:
        out.write(str(F1_max_score))


if __name__ == "__main__":
    main()