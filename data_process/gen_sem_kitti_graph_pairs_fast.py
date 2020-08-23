import numpy as np
import argparse
import os
import json
import pykitti
import random
from tqdm import tqdm

'''
input: graph_dir, random drop out negative samples?
output: graph_pairs
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', '-d', type=str, required=False, default="/media/yxm/文档/data/kitti/dataset/", help='Dataset path.')
    parser.add_argument('--graph_dir', '-g', type=str, required=False, default="", help='Graph path.')
    parser.add_argument('--output_dir', '-o', type=str, required=False, default="", help='Output path.')
    parser.add_argument('--pair_list', '-p', type=str, required=False, default="/pair_list", help='Output path.')
    parser.add_argument('--pos_thre', type=int, required=False, default=3, help='Positive threshold.')
    parser.add_argument('--neg_thre', type=int, required=False, default=20, help='Negative threshold.')

    args, unparsed = parser.parse_known_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sequences = ["00","01","02","03","04","05","06","07","08","09","10"]

    for sq in sequences[:]:
        sq_graph_dir = os.path.join(args.graph_dir, sq)
        graph_pair_list = np.load(args.test_list_dir+'pair_list_3_20_'+str(sq)+'.npy')
        # print(graph_pair_list.shape)
        dataset = pykitti.odometry(args.dataset_dir, sq)
        output_dir_sq = os.path.join(args.output_dir, sq)

        if not os.path.exists(output_dir_sq):
            os.makedirs(output_dir_sq)
        for i in tqdm(range(len(graph_pair_list))):
            [id_i, id_j] = (graph_pair_list[i].split('.')[0]).split('_')
            # print(id_i, id_j)
            graph_i = json.load(open(os.path.join(sq_graph_dir, id_i.zfill(6)+'.json')))
            graph_j = json.load(open(os.path.join(sq_graph_dir, id_j.zfill(6)+'.json')))

            graph_i["nodes_1"] = graph_i.pop("nodes")
            graph_i["edges_1"] = graph_i.pop("edges")
            graph_i["weights_1"] = graph_i.pop("weights")
            graph_i["centers_1"] = graph_i.pop("centers")
            pose_i = dataset.poses[int(id_i)]

            graph_j["nodes_2"] = graph_j.pop("nodes")
            graph_j["edges_2"] = graph_j.pop("edges")
            graph_j["weights_2"] = graph_j.pop("weights")
            graph_j["centers_2"] = graph_j.pop("centers")
            pose_j = dataset.poses[int(id_j)]

            graph_pair = dict(graph_i, **graph_j)

            dist = np.linalg.norm(pose_i[0::2, -1] - pose_j[0::2, -1])
            graph_pair.update({"distance": dist})
            # print(graph_pair)

            # write out graph as json file
            file_name = output_dir_sq + "/" + str(id_i) + "_" + str(id_j) + ".json"
            # print("output json: ", file_name)
            with open(file_name, "w", encoding="utf-8") as file:
                json.dump(graph_pair, file)


