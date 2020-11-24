import yaml
import os
class sgpr_args():
    def __init__(self):
        #common
        self.model=""
        self.graph_pairs_dir="/dir_of_graph_pairs"
        self.p_thresh=3
        self.batch_size=128
        #arch
        self.keep_node=1
        self.filters_1=64
        self.filters_2=64
        self.filters_3=32
        self.tensor_neurons=16
        self.bottle_neck_neurons=16
        #train
        self.epochs=500
        self.fold="00"
        self.dropout=0
        self.learning_rate=1e-3
        self.weight_decay=5e-4
        self.gpu=0
        self.logdir="./logs"
        self.node_num=100
        #eva_batch
        self.sequences=[]
        self.output_path="./eva"
        #eva_pair
        self.pair_file=""

    def load(self,config_file):
        config_args=yaml.load(open(os.path.abspath(config_file)))
        #arch
        self.keep_node=config_args['arch']['keep_node']
        self.filters_1=config_args['arch']['filters_1']
        self.filters_2=config_args['arch']['filters_2']
        self.filters_3=config_args['arch']['filters_3']
        self.tensor_neurons=config_args['arch']['tensor_neurons']
        self.bottle_neck_neurons=config_args['arch']['bottle_neck_neurons']
        #train
        self.epochs=config_args['train']['epochs']
        self.fold=config_args['train']['fold']
        self.dropout=config_args['train']['dropout']
        self.learning_rate=config_args['train']['learning_rate']
        self.weight_decay=config_args['train']['weight_decay']
        self.gpu=config_args['train']['gpu']
        self.logdir=config_args['train']['logdir']
        self.node_num=config_args['train']['node_num']
        #common
        self.model=config_args['common']['model']
        self.batch_size=config_args['common']['batch_size']
        self.p_thresh=config_args['common']['p_thresh']
        self.graph_pairs_dir=config_args['common']['graph_pairs_dir']
        #eva_batch
        self.sequences=config_args['eva_batch']['sequences']
        self.output_path=config_args['eva_batch']['output_path']
        #eva_pair
        self.pair_file=config_args['eva_pair']['pair_file']