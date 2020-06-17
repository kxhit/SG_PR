import numpy as np
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
    dataset_dir = "/media/yxm/文档/data/kitti/dataset"
    graph_dir = "/media/kx/Semantic_KITTI/darkne53_graph_ds" # edge 3m
    output_dir = "/media/kx/Semantic_KITTI/graph_pairs_darknet53_10_20_ds"
    test_list_dir = "/media/kx/Semantic_KITTI/graph_pairs_sem_10_20_ds/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sequences = ["00","01","02","03","04","05","06","07","08","09","10"]

    for sq in sequences[0:2]:
        print(sq)
        sq_graph_dir = os.path.join(graph_dir, sq)
        # graph_list = os.listdir(sq_graph_dir)
        # graph_list.sort()
        graph_pair_list = np.load(test_list_dir+'pair_list_10_20_'+str(sq)+'.npy')
        print(graph_pair_list.shape)
        dataset = pykitti.odometry(dataset_dir, sq)
        output_dir_sq = os.path.join(output_dir, sq)

        if not os.path.exists(output_dir_sq):
            os.makedirs(output_dir_sq)
        # cnt = 0
        for i in tqdm(range(len(graph_pair_list))):

            [id_i, id_j] = (graph_pair_list[i].split('.')[0]).split('_')
            # print(id_i, id_j)
            graph_i = json.load(open(os.path.join(sq_graph_dir, id_i.zfill(6)+'.json')))
            graph_j = json.load(open(os.path.join(sq_graph_dir, id_j.zfill(6)+'.json')))

            # if len(graph_i["nodes"]) <= 5: # too few clusters
            #     continue
            graph_i["nodes_1"] = graph_i.pop("nodes")
            graph_i["edges_1"] = graph_i.pop("edges")
            graph_i["weights_1"] = graph_i.pop("weights")
            graph_i["centers_1"] = graph_i.pop("centers")
            pose_i = dataset.poses[int(id_i)]
            # print('pose_i:',pose_i)

            # if len(graph_j["nodes"]) <= 5:  # too few clusters
            #     continue
            graph_j["nodes_2"] = graph_j.pop("nodes")
            graph_j["edges_2"] = graph_j.pop("edges")
            graph_j["weights_2"] = graph_j.pop("weights")
            graph_j["centers_2"] = graph_j.pop("centers")
            pose_j = dataset.poses[int(id_j)]
            # print('pose_j:',pose_j)
            graph_pair = dict(graph_i, **graph_j)

            dist = np.linalg.norm(pose_i[0::2, -1] - pose_j[0::2, -1])
            graph_pair.update({"distance": dist})
            # print(graph_pair)

            # write out graph as json file
            file_name = output_dir_sq + "/" + str(id_i) + "_" + str(id_j) + ".json"
            # print("output json: ", file_name)
            with open(file_name, "w", encoding="utf-8") as file:
                json.dump(graph_pair, file)


