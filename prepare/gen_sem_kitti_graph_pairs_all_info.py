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
    graph_dir = "/media/kx/Semantic_KITTI/sem_kitti_graph_ds/" # edge 5m
    output_dir = "/media/kx/Semantic_KITTI/graph_pairs_sem_10_20_ds"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sequences = ["00","01","02","03","04","05","06","07","08","09","10"]

    for sq in sequences[7:]:
        print(sq)
        sq_graph_dir = os.path.join(graph_dir, sq)
        graph_list = os.listdir(sq_graph_dir)
        graph_list.sort()
        dataset = pykitti.odometry(dataset_dir, sq)
        output_dir_sq = os.path.join(output_dir, sq)
        if not os.path.exists(output_dir_sq):
            os.makedirs(output_dir_sq)
        # cnt = 0
        for i in tqdm(range(len(graph_list))):
            graph_i = json.load(open(os.path.join(sq_graph_dir, graph_list[i])))
            # print(graph_i)
            if len(graph_i["edges"]) <= 5: # too few clusters
                continue
            graph_i["nodes_1"] = graph_i.pop("nodes")
            graph_i["edges_1"] = graph_i.pop("edges")
            graph_i["weights_1"] = graph_i.pop("weights")
            graph_i["centers_1"] = graph_i.pop("centers")
            pose_i = dataset.poses[i]
            for j in range(i, len(graph_list)):
                graph_j = json.load(open(os.path.join(sq_graph_dir, graph_list[j])))
                if len(graph_j["edges"]) <= 5:  # too few clusters
                    continue
                graph_j["nodes_2"] = graph_j.pop("nodes")
                graph_j["edges_2"] = graph_j.pop("edges")
                graph_j["weights_2"] = graph_j.pop("weights")
                graph_j["centers_2"] = graph_j.pop("centers")
                graph_pair = dict(graph_i, **graph_j)

                pose_j = dataset.poses[j]
                dist = np.linalg.norm(pose_i[0::2, -1] - pose_j[0::2, -1])

                choose_prob = False

                # sampling strategy
                if dist >=10 and dist <= 20:
                    continue

                if dist <=10: # dist < 3m choose prob = 1
                    if random.random() <= 0.5:
                        choose_prob = True
                    # else:
                    #     choose_prob = False

                else:   # dist > 20m choose prob = 0.5%
                    if random.random() <= 0.005:
                        choose_prob = True
                    else:
                        choose_prob = False

                if choose_prob == True:
                    graph_pair.update({"distance": dist})

                    # write out graph as json file
                    file_name = output_dir_sq + "/" + str(i) + "_" + str(j) + ".json"
                    # print("output json: ", file_name)
                    with open(file_name, "w", encoding="utf-8") as file:
                        json.dump(graph_pair, file)



