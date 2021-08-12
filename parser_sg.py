import yaml
import os
class sgpr_args():
    def __init__(self):
        #common
        self.model=""
        self.graph_pairs_dir="/dir_of_graph_pairs"
        self.p_thresh=3
        self.batch_size=128
        self.pair_list_dir=''
        #arch
        self.keep_node=1
        self.filters_1=64
        self.filters_2=64
        self.filters_3=32
        self.tensor_neurons=16
        self.bottle_neck_neurons=16
        self.K=10
        #train
        self.epochs=500
        self.train_sequences=[]
        self.eval_sequences=[]
        self.dropout=0
        self.learning_rate=1e-3
        self.weight_decay=5e-4
        self.gpu=0
        self.logdir="./logs"
        self.node_num=100
        #eva_batch
        self.sequences=[]
        self.output_path="./eva"
        self.show=False
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
        self.K=config_args['arch']['K']
        #train
        self.epochs=config_args['train']['epochs']
        self.train_sequences=config_args['train']['train_sequences']
        self.eval_sequences=config_args['train']['eval_sequences']
        self.dropout=config_args['train']['dropout']
        self.learning_rate=config_args['train']['learning_rate']
        self.weight_decay=config_args['train']['weight_decay']
        self.gpu=config_args['train']['gpu']
        self.logdir=config_args['train']['logdir']
        self.node_num=config_args['train']['node_num']
        #common
        self.model=config_args['common']['model']
        self.cuda=config_args['common']['cuda']
        self.batch_size=config_args['common']['batch_size']
        self.p_thresh=config_args['common']['p_thresh']
        self.graph_pairs_dir=config_args['common']['graph_pairs_dir']
        self.pair_list_dir=config_args['common']['pair_list_dir']
        #eva_batch
        self.sequences=config_args['eva_batch']['sequences']
        self.output_path=config_args['eva_batch']['output_path']
        self.show=config_args['eva_batch']['show']
        #eva_pair
        self.pair_file=config_args['eva_pair']['pair_file']