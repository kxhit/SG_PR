from utils import tab_printer
from sg_net import SGTrainer
from parser_sg import sgpr_args
import numpy as np
from tqdm import tqdm
import os
import json
from matplotlib import pyplot as plt
from sklearn import metrics
import torch
import time


def main():
    args = sgpr_args()
    args.load('./config/config.yml')
    tab_printer(args)
    trainer = SGTrainer(args, args.model)
    trainer.model.eval()
    pred, gt = trainer.eval_batch_pair([args.pair_file,])
    print("Score:",pred[0])


if __name__ == "__main__":
    main()