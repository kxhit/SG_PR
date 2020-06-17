import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import os

def pr_roc_metrics(output_path, sequence, pred_db_list, gt_db, method_list):

    color_method = {"M2DP": 'darkorange',
                    "SC": 'blue',
                    "PV_PRE": 'purple',
                    "PV_KITTI": 'pink',
                    "Ours_RN": 'magenta',
                    "Ours_SK": 'red'}
    lw = 2
    # plot ROC Curve
    plt.figure(0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title(sequence + ' ROC Curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') # baseline

    roc_out = output_path + "/" + sequence + "_all_roc_curve.png"
    for i in range(len(method_list)):
        method = method_list[i]
        pred_db = pred_db_list[i]
        #####ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(gt_db, pred_db)
        roc_auc = metrics.auc(fpr, tpr)
        # print("fpr: ", fpr)
        # print("tpr: ", tpr)
        # print("thresholds: ", roc_thresholds)
        # print("roc_auc: ", roc_auc)

        plt.plot(fpr, tpr, color=color_method[method],
                     lw=lw, label= method + ' (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线

    plt.legend(loc="lower right")
    plt.savefig(roc_out)
    # plt.show()
    plt.close()




    #### P-R
    # plot p-r curve
    plt.figure(1)
    lw = 2
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    # plt.title(sequence + ' Precision-Recall Curve')

    pr_out = output_path + "/" + sequence + "_all_pr_curve.png"
    for i in range(len(method_list)):
        method = method_list[i]
        pred_db = pred_db_list[i]
        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt_db, pred_db)

        plt.plot(recall, precision, color=color_method[method],
                lw=lw, label= method)  ###假正率为横坐标，真正率为纵坐标做曲线

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # plt.legend(loc="lower right")
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(pr_out)
    # plt.show()
    plt.close()

drop = "drop0"
threshold = "10_20"
# sequences = ["00", "02", "05", "08"]
sequences = ["01", "02"]
# sequences = ["00"]
root_dir = "Ford/all/"
# root_dir = "/home/kx/project/3D/3D_loop_closure_detection/3D_dl_lc/eval/ALL/1-fold_drop30/rangenet/" + drop + "/"

for sq in sequences:
    output_dir = os.path.join(root_dir, threshold, sq)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)
    # load M2DP
    M2DP_base_dir = "compare/M2DP/eval/Ford/"+threshold+"/"
    M2DP_pred_db = M2DP_base_dir + sq + "_M2DP_db.npy"
    M2DP_gt_db = M2DP_base_dir  + sq + "_gt_db.npy"
    M2DP_pred = np.load(M2DP_pred_db)
    M2DP_gt = np.load(M2DP_gt_db)

    # load SC
    SC_base_dir = "compare/SC/eval/Ford/"+threshold+"/"
    SC_pred_db = SC_base_dir + sq + "_SC_db.npy"
    SC_gt_db = SC_base_dir  + sq + "_gt_db.npy"
    SC_pred = np.load(SC_pred_db)
    SC_gt = np.load(SC_gt_db)

    # todo: load PointNetVLAD
    # load PointNetVLAD pretrained refined model provided by author
    PV_ref_base_dir = "compare/PV_PRE/eval/Ford/"+threshold+"/"
    PV_ref_pred_db = PV_ref_base_dir + sq + "_PV_db.npy"
    PV_ref_gt_db = PV_ref_base_dir + sq + "_gt_db.npy"
    PV_ref_pred = np.load(PV_ref_pred_db)
    PV_ref_gt = np.load(PV_ref_gt_db)

    # load PointNetVLAD trained on KITTI 1-fold
    PV_base_dir = "compare/PV_KITTI/eval/Ford/"+threshold+"/"
    PV_pred_db = PV_base_dir + sq + "_PV_db.npy"
    PV_gt_db = PV_base_dir + sq + "_gt_db.npy"
    PV_pred = np.load(PV_pred_db)
    PV_gt = np.load(PV_gt_db)

    # # load SK
    # simgnn_base_dir = threshold + "/Ours_SK/" + drop + "/"
    # simgnn_pred_db = simgnn_base_dir + sq + "/" + sq + "_DL_db.npy"
    # simgnn_gt_db = simgnn_base_dir + sq + "/" + sq + "_gt_db.npy"
    # simgnn_pred = np.load(simgnn_pred_db)
    # simgnn_gt = np.load(simgnn_gt_db)

    # load RN
    rangenet_base_dir = "Ford/"+ threshold + "/"+sq+"/Ours_RN/" + drop + "/"
    rangenet_pred_db = rangenet_base_dir + sq + "_DL_db.npy"
    rangenet_gt_db = rangenet_base_dir + sq + "_gt_db.npy"
    rangenet_pred = np.load(rangenet_pred_db)
    rangenet_gt = np.load(rangenet_gt_db)

    # assert (M2DP_gt == SC_gt).all()
    # assert (SC_gt == simgnn_gt).all()
    # assert (SC_gt == PV_gt).all()

    pred_list = []
    method_list = []
    pred_list.append(M2DP_pred)
    method_list.append("M2DP")
    pred_list.append(SC_pred)
    method_list.append("SC")
    pred_list.append(PV_ref_pred)
    method_list.append("PV_PRE")
    pred_list.append(PV_pred)
    method_list.append("PV_KITTI")
    pred_list.append(rangenet_pred)
    method_list.append("Ours_RN")
    # pred_list.append(simgnn_pred)
    # method_list.append("Ours_SK")

    # metrics.roc_curve(M2DP_gt, M2DP_pred)

    pr_roc_metrics(output_dir, sq, pred_list, SC_gt, method_list)