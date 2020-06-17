from utils import tab_printer
from sg_net import SGTrainer
from parser_sg import parameter_parser

def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = SGTrainer(args)
    trainer.fit()
    trainer.score()
    
if __name__ == "__main__":
    main()


# bash command

# python main_sg.py --p_thresh 10 --graph_pairs_dir /media/work/data/kitti/odometry/semantic-kitti/10_20/graph_pairs_sem_10_20_ds --logdir 10_20/log_10_20_fold00 --fold 00