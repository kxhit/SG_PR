import numpy as np
import os
import json
from tqdm import tqdm
import pykitti
from gen_gt import t_dist

def listDir(path, list_name):
    """
    :param path: root_dir
    :param list_name: abs paths of all files under the root_dir
    :return:
    """
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listDir(file_path, list_name)
        else:
            list_name.append(file_path)

def sample_data(graph_list, dataset):
    new_list = []
    for sample in tqdm(graph_list):
        data = json.load(open(sample))
        if data["distance"] > 10 and data["distance"] < 20: #　TODO change the threshold
            continue
        # print(sample)
        sample_base = sample.split('/')[-1]
        [i, j] = sample_base.split('.')[0].split('_')
        if not t_dist(dataset.timestamps[int(i)], dataset.timestamps[int(j)], threshold=30):
            continue
        else:
            new_list.append(sample_base)

    return new_list

sequences = ["00","02","05","08","06","07"]
for sq in sequences[0:]:
    dataset = pykitti.odometry('/media/yxm/文档/data/kitti/dataset/', sq)
    print("sequence: ", sq)
    args_testing_graphs = "/media/kx/Semantic_KITTI/graph_pairs_sem_10_20_ds/" + sq
    testing_graphs = []
    listDir(args_testing_graphs, testing_graphs)

    new_test = sample_data(testing_graphs, dataset)
    print("length", len(new_test))
    output = "/media/kx/Semantic_KITTI/graph_pairs_sem_10_20_ds/draw_curve_" + sq + ".npy"
    np.save(output, np.asarray(new_test))

