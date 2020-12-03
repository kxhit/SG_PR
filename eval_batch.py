from numpy.core.arrayprint import printoptions
from utils import tab_printer
from sg_net import SGTrainer
from parser_sg import sgpr_args
import numpy as np
from tqdm import tqdm
import os
import sys
from matplotlib import pyplot as plt
from sklearn import metrics
from utils import *


def main():
    args = sgpr_args()
    if len(sys.argv)>1:
        args.load(sys.argv[1])
    else:
        args.load('./config/config.yml')
    args.load(os.path.abspath('./config/config.yml'))
    tab_printer(args)
    trainer = SGTrainer(args, False)
    trainer.model.eval()
    if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
    for sequence in tqdm(args.sequences):
        print("sequence: ", sequence)
        gt_db = []
        pred_db = []
        graph_pairs=load_paires(os.path.join(args.pair_list_dir, sequence+".txt"),args.graph_pairs_dir)
        batches = [graph_pairs[graph:graph + args.batch_size] for graph in
                   range(0, len(graph_pairs), args.batch_size)]
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
        gt_db_path = os.path.join(args.output_path,sequence + "_gt_db.npy")
        pred_db_path = os.path.join(args.output_path,sequence + "_DL_db.npy")
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
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('DL ROC Curve')
        plt.legend(loc="lower right")
        roc_out = os.path.join(args.output_path, sequence + "_DL_roc_curve.png")
        plt.savefig(roc_out)

        #### P-R
        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt_db, pred_db)
        # plot p-r curve
        plt.figure(1)
        lw = 2
        plt.plot(recall, precision, color='darkorange',
                 lw=lw, label='P-R curve') 
        plt.axis([0,1,0,1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('DL Precision-Recall Curve')
        plt.legend(loc="lower right")
        pr_out = os.path.join(args.output_path, sequence + "_DL_pr_curve.png")
        plt.savefig(pr_out)
        if args.show:
            plt.show()
        # calc F1-score
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        F1_max_score = np.max(F1_score)
        f1_out = os.path.join(args.output_path, sequence + "_DL_F1_max.txt")
        print('F1 max score',F1_max_score)
        with open(f1_out, "w") as out:
            out.write(str(F1_max_score))


if __name__ == "__main__":
    main()