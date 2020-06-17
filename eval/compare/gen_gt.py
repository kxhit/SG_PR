import numpy as np
import pykitti
import json
import os
from tqdm import tqdm

def get_dataset(sequence_id='00', basedir= '/media/datasets/kx/kitti_odometry/dataset/'):
    return pykitti.odometry(basedir, sequence_id)


def p_dist(pose1, pose2, threshold=3, print_bool=False):
    # dist = np.linalg.norm(pose1[:,-1]-pose2[:,-1])    # xyz
    dist = np.linalg.norm(pose1[0::2, -1] - pose2[0::2, -1])  # xz in cam0
    if print_bool==True:
        print(dist)
    if abs(dist) <= threshold:
        return True
    else:
        return False

def t_dist(t1, t2, threshold=10):
    if abs((t1-t2).total_seconds()) >= threshold:
        return True
    else:
        return False


def train_test_split(train_nums=5, on_framenum=True,
                     ids=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']):
    ids = np.array(ids)
    if on_framenum:
        # choose training dataset with fewer frame nums
        framenums = [len(get_dataset(id).timestamps) for id in ids]
        print(framenums)

        train_ids = list(ids[np.argsort(framenums)[:train_nums]])
        test_ids = list(ids[np.argsort(framenums)[train_nums:]])
        return train_ids, test_ids

    else:
        train_ids = list(ids[:train_nums])
        test_ids = list(ids[train_nums:])
        return train_ids, test_ids




def get_positive_dict(ids, output_dir, d_thresh, t_thresh):
    positive = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for id in tqdm(ids):
        dataset = get_dataset(id)

        if id not in positive:
            positive[id] = {}

        for t1 in range(len(dataset.timestamps)):
            # for t2 in range(t1, len(dataset.timestamps)):
            for t2 in range(len(dataset.timestamps)):
                if p_dist(dataset.poses[t1], dataset.poses[t2], d_thresh) & t_dist(dataset.timestamps[t1], dataset.timestamps[t2], t_thresh):
                    if t1 not in positive[id]:
                        positive[id][t1] = []
                    positive[id][t1].append(t2)

    with open('{}/positive_sequence_D-{}_T-{}.json'.format(output_dir, d_thresh, t_thresh), 'w') as f:
        json.dump(positive, f)

    return positive



if __name__ == '__main__':
    # train_ids, test_ids = train_test_split()
    # train_positive_dt = get_positive_dict(train_ids, '../train', d_thresh=5, t_thresh=10)
    # test_positive_dt = get_positive_dict(test_ids, '../test', d_thresh=5, t_thresh=10)
    # train_positive_d = get_positive_dict(train_ids, '../train', d_thresh=5, t_thresh=0)
    # test_positive_d = get_positive_dict(test_ids, '../test', d_thresh=5, t_thresh=0)
    # train_ids = ["01","03","04","06","07","09","10"]
    # test_ids = ["00", "02", "05", "08"]
    all_sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    # train_positive_dt = get_positive_dict(train_ids, '../train', d_thresh=25, t_thresh=50)
    # test_positive_dt = get_positive_dict(test_ids, '../semantic/test', d_thresh=3, t_thresh=50)
    # train_positive_d = get_positive_dict(train_ids, '../semantic/train', d_thresh=3, t_thresh=0)
    # # test_positive_d = get_positive_dict(test_ids, '../test', d_thresh=25, t_thresh=0)
    # test_positive_dt = get_positive_dict(test_ids, '../semantic/test', d_thresh=20, t_thresh=0)
    # train_positive_d = get_positive_dict(train_ids, '../semantic/train', d_thresh=20, t_thresh=0)

    # get_positive_dict(all_sequences, '../PV/KITTI_all', d_thresh=3, t_thresh=0)
    # get_positive_dict(all_sequences, '../PV/KITTI_all', d_thresh=20, t_thresh=0)
    get_positive_dict(all_sequences, '../yxm/KITTI_2', d_thresh=3, t_thresh=30)
    get_positive_dict(all_sequences, '../yxm/KITTI_2', d_thresh=20, t_thresh=0)

