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
    dist_thresh = 10
    Ford = "01"
    # Ford 02
    if Ford == "02":
        dataset_dir = "/media/work/data/Ford/IJRR-Dataset-2/SG"
        graph_dir = "/media/work/data/Ford/IJRR-Dataset-2/sem_graph_custom/22" # graph dir
        output_dir = "/media/work/data/Ford/IJRR-Dataset-2/SG/" + str(dist_thresh) + "_20/graph_paris"
        draw_curve = os.path.join("/media/work/data/Ford/IJRR-Dataset-2/SG/" + str(dist_thresh) + "_20","draw_curve.npy")
    # Ford 01
    else:
        dataset_dir = "/media/work/data/Ford/IJRR-Dataset-1/SG"
        graph_dir = "/media/work/data/Ford/IJRR-Dataset-1/SG/sem_graphs/"  # graph dir
        output_dir = "/media/work/data/Ford/IJRR-Dataset-1/SG/" + str(dist_thresh) + "_20/graph_paris"
        draw_curve = os.path.join("/media/work/data/Ford/IJRR-Dataset-1/SG/" + str(dist_thresh) + "_20", "draw_curve.npy")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    graph_list = os.listdir(graph_dir)
    graph_list.sort()
    # dataset = pykitti.odometry(dataset_dir, sq)
    # load ford dataset pcs poses
    poses = np.fromfile(os.path.join(dataset_dir,"poses_wv.bin"),dtype=np.float32).reshape(-1,6)
    times = np.fromfile(os.path.join(dataset_dir,"timestamps.bin")).reshape(-1,1)
    test_list = []
    true_positive_cnt = 0

    for i in tqdm(range(len(graph_list))):
        graph_i = json.load(open(os.path.join(graph_dir, graph_list[i])))
        # print(graph_i)
        if len(graph_i["nodes"]) <= 5: # too few clusters
            continue
        graph_i["nodes_1"] = graph_i.pop("nodes")
        graph_i["centers_1"] = graph_i.pop("centers")
        pose_i = poses[i]
        for j in range(i, len(graph_list)):
            graph_j = json.load(open(os.path.join(graph_dir, graph_list[j])))
            if len(graph_j["nodes"]) <= 5:  # too few clusters
                continue
            graph_j["nodes_2"] = graph_j.pop("nodes")
            graph_j["centers_2"] = graph_j.pop("centers")
            graph_pair = dict(graph_i, **graph_j)

            pose_j = poses[j]
            dist = np.linalg.norm(pose_i[0:3] - pose_j[0:3])

            choose_prob = False
            if dist_thresh == 3:
                prob = 1
            else:
                prob = 0.5
            # sampling strategy
            if dist >=dist_thresh and dist <= 20:
                continue

            if dist <=dist_thresh: # dist < 3m choose prob = 1
                if random.random() <= prob:    # todo
                    choose_prob = True
                # else:
                #     choose_prob = False

            else:   # dist > 20m choose prob = 0.5%
                if random.random() <= 0.005:
                    choose_prob = True
                else:
                    choose_prob = False

            if choose_prob == True:
                graph_pair.update({"distance": float(dist)})

                # write out graph as json file
                if Ford == "01":
                    file_name = output_dir + "/" + str(i + 75) + "_" + str(j + 75) + ".json"
                else:
                    file_name = output_dir + "/" + str(i+1) + "_" + str(j+1) + ".json"
                # print("output json: ", file_name)
                with open(file_name, "w", encoding="utf-8") as file:
                    json.dump(graph_pair, file)

                # whether count as test pairs
                if graph_pair["distance"] > dist_thresh and graph_pair["distance"] < 20:
                    continue

                if abs(times[i]-times[j])*1e-6 < 30:    # 30 seconds time constrains for filtering out easy positive samples
                    continue
                if Ford == "01":
                    test_list.append(str(i + 75) + "_" + str(j + 75) + ".json")
                else:
                    test_list.append(str(i+1) + "_" + str(j+1) + ".json")
                if graph_pair["distance"] < dist_thresh:
                    true_positive_cnt += 1
                    # print("TP: ", file)
                    print("cnt: ", true_positive_cnt)

    np.save(draw_curve, np.asarray(test_list))



