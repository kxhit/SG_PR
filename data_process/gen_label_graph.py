#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import numpy as np
import pcl
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from auxiliary.laserscan import SemLaserScan
import random
import open3d
# from open3d import *
import json
import collections
from tqdm import tqdm
import time
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

learning_map={0 : 0,     # "unlabeled"
    1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped       x
    10: 1,     # "car"
    11: 2,     # "bicycle"                                                              x
    13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,     # "motorcycle"                                                           x
    16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"                                                               x
    31: 7,     # "bicyclist"                                                            x
    32: 8,     # "motorcyclist"                                                         x
    40: 9,     # "road"
    44: 10,     # "parking"  
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,    # "moving-person" to "person" ------------------------------mapped
    255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,    # "moving-truck" to "truck" --------------------------------mapped
    259: 5}

max_key = max(learning_map.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(learning_map.keys())] = list(learning_map.values())

node_map={
  1: 0,      # "car"
  4: 1,      # "truck"
  5: 2,      # "other-vehicle"
  11: 3,     # "sidewalk"
  12: 4,     # "other-ground"
  13: 5,     # "building"
  14: 6,     # "fence"
  15: 7,     # "vegetation"
  16: 8,     # "trunk"
  17: 9,     # "terrain"
  18: 10,    # "pole"
  19: 11     # "traffic-sign"
  }

def open3d_color():
    i = random.random()
    j = random.random()
    k = random.random()
    return (i,j,k)

def Visualize():
    viz_point = open3d.PointCloud()
    point_cloud = open3d.PointCloud()
    
    for id_i, label_i in enumerate(sem_label_set):
        print('sem_label:', label_i)
    
        index = np.argwhere(sem_label == label_i)
        index = index.reshape(index.shape[0])
        sem_cluster = points[index, :]
    
        point_cloud.points = open3d.Vector3dVector(sem_cluster[:, 0:3])
        color = color_map[learning_map_inv[label_i]]
        color = (color[0] / 255, color[1] / 255, color[2] / 255)
        # print(color)
        point_cloud.paint_uniform_color(color)
        viz_point += point_cloud
    
        open3d.draw_geometries([point_cloud], window_name='semantic label:' + str(111),
                           width=1920, height=1080, left=50, top=50)

def _make_point_field(num_field):
    msg_pf1 = pc2.PointField()
    msg_pf1.name = np.str('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)

    msg_pf2 = pc2.PointField()
    msg_pf2.name = np.str('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)

    msg_pf3 = pc2.PointField()
    msg_pf3.name = np.str('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)

    if num_field == 4:
        msg_pf4 = pc2.PointField()
        msg_pf4.name = np.str('node')
        msg_pf4.offset = np.uint32(16)
        msg_pf4.datatype = np.uint8(7)
        msg_pf4.count = np.uint32(1)
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]

    elif num_field ==6:
        msg_pf4 = pc2.PointField()
        msg_pf4.name = np.str('intensity')
        msg_pf4.offset = np.uint32(16)
        msg_pf4.datatype = np.uint8(7)  #float64
        msg_pf4.count = np.uint32(1)

        msg_pf5 = pc2.PointField()
        msg_pf5.name = np.str('sem_label')
        msg_pf5.offset = np.uint32(20)
        msg_pf5.datatype = np.uint8(7)  # 4 int16
        msg_pf5.count = np.uint32(1)

        msg_pf6 = pc2.PointField()
        msg_pf6.name = np.str('inst_label')
        msg_pf6.offset = np.uint32(24)
        msg_pf6.datatype = np.uint8(7)  # 4 int16
        msg_pf6.count = np.uint32(1)

        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5, msg_pf6]
    # if num_field == 4:
    #     fields = [PointField('x', 0, PointField.FLOAT32, 1),
    #       PointField('y', 4, PointField.FLOAT32, 1),
    #       PointField('z', 8, PointField.FLOAT32, 1),
    #       PointField('node', 16, PointField.UINT32, 1),
    #       ]
    #     return fields
    # elif num_field == 6:
    #     fields = [PointField('x', 0, PointField.FLOAT32, 1),
    #       PointField('y', 4, PointField.FLOAT32, 1),
    #       PointField('z', 8, PointField.FLOAT32, 1),
    #       PointField('intensity', 12, PointField.FLOAT32, 1),
    #       PointField('sem_label', 16, PointField.UINT32, 1),
    #       PointField('inst_label', 20, PointField.UINT32, 1),
    #       ]

    #     return fields
    else:
        raise ValueError("wrong num_field.")


class Semantic_kitti_node(object):
    def __init__(self,  pub_rate=10, label_topic='', graph_topic=''):
        """
        ros node spin in init function
        :param pub_rate:
        :param pub_topic:
        """
        self._pub_rate = pub_rate
        # publisher
        self._labels_pub = rospy.Publisher(label_topic, PointCloud2, queue_size=10)
        self._graph_pub = rospy.Publisher(graph_topic, PointCloud2, queue_size = 10)
        # ros node init
        rospy.init_node('node', anonymous=True)
        rospy.loginfo("node started.")

        self.header1 = Header()
        self.header1.stamp = rospy.Time()
        self.header1.frame_id = "velodyne"

        self.header2 = Header()
        self.header2.stamp = rospy.Time()
        self.header2.frame_id = "velodyne"
    
    def gen_labels(self, FLAGS, scan_name, label_name, label_output_dir):
        # start = time.time()
        # open scan
        # TODO(yxm): downsampling
        scan = np.fromfile(scan_name, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        # put in attribute
        points = scan[:, 0:4]  # get xyzr
        remissions = scan[:, 3]  # get remission

        label = np.fromfile(label_name, dtype=np.uint32)
        label = label.reshape((-1))

        # demolition or not
        if FLAGS.demolition == True:
            start_angle = np.random.random()
            start_angle *= 360
            end_angle = (start_angle + drop_angle)%360

            angle = np.arctan2(points[:, 1], points[:, 0])
            angle = angle*180/np.pi
            angle += 180
            # print("angle:", angle)
            if end_angle > start_angle:
                remain_id = np.argwhere(angle < start_angle).reshape(-1)
                remain_id = np.append(remain_id, np.argwhere(angle > end_angle).reshape(-1))
            else:
                remain_id = np.argwhere((angle > end_angle) & (angle < start_angle)).reshape(-1)

            points = points[remain_id, : ]
            label = label[remain_id]

        if label.shape[0] == points.shape[0]:
            sem_label = label & 0xFFFF  # semantic label in lower half
            inst_label = label >> 16  # instance id in upper half
            assert ((sem_label + (inst_label << 16) == label).all())
        else:
            print("Points shape: ", points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        sem_label = remap_lut[sem_label]
        sem_label_set = list(set(sem_label))
        sem_label_set.sort()

        # Start clustering
        cluster = []
        inst_id = 0
        for id_i, label_i in enumerate(sem_label_set):
            # print('sem_label:', label_i)
            index = np.argwhere(sem_label == label_i)
            index = index.reshape(index.shape[0])
            sem_cluster = points[index, :]
            # print("sem_cluster_shape:",sem_cluster.shape[0])

            tmp_inst_label = inst_label[index]
            tmp_inst_set = list(set(tmp_inst_label))
            tmp_inst_set.sort()
            # print(tmp_inst_set)

            if label_i in [9, 10]:    # road/parking, dont need to cluster
                inst_cluster = sem_cluster
                inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), label_i, dtype=np.uint32)), axis=1)
                # inst_cluster = np.insert(inst_cluster, 4, label_i, axis=1)
                inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), inst_id, dtype=np.uint32)), axis=1)
                inst_id = inst_id + 1
                cluster.extend(inst_cluster)  # Nx6                
                continue
                
            elif label_i in [0,2,3,6,7,8]:    # discard
                continue
            
            elif len(tmp_inst_set) > 1 or (len(tmp_inst_set) == 1 and tmp_inst_set[0] != 0):     # have instance labels
                for id_j, label_j in enumerate(tmp_inst_set):
                    points_index = np.argwhere(tmp_inst_label == label_j)
                    points_index = points_index.reshape(points_index.shape[0])
                    # print(id_j, 'inst_size:', len(points_index))
                    if len(points_index) <= 20:
                        continue
                    inst_cluster = sem_cluster[points_index, :]
                    inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), label_i, dtype=np.uint32)), axis=1)
                    # inst_cluster = np.insert(inst_cluster, 4, label_i, axis=1)
                    inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), inst_id, dtype=np.uint32)), axis=1)
                    inst_id = inst_id + 1
                    cluster.extend(inst_cluster)
            else:    # Euclidean cluster
                # time_start = time.time()
                if label_i in [1, 4, 5, 14]:     # car truck other-vehicle fence
                    cluster_tolerance = 0.5
                elif label_i in [11, 12, 13, 15, 17]:    # sidewalk other-ground building vegetation terrain
                    cluster_tolerance = 2
                else:
                    cluster_tolerance = 0.2

                if label_i in [16, 19]:    # trunk traffic-sign
                    min_size = 50
                elif label_i == 15:     # vegetation
                    min_size = 200
                elif label_i in [11, 12, 13, 17]:    # sidewalk other-ground building terrain
                    min_size = 300
                else:
                    min_size = 100

                # print(cluster_tolerance, min_size)
                cloud = pcl.PointCloud(sem_cluster[:, 0:3])
                tree = cloud.make_kdtree()
                ec = cloud.make_EuclideanClusterExtraction()
                ec.set_ClusterTolerance(cluster_tolerance)
                ec.set_MinClusterSize(min_size)
                ec.set_MaxClusterSize(50000)
                ec.set_SearchMethod(tree)
                cluster_indices = ec.Extract()
                # time_end = time.time()
                # print(time_end - time_start)
                for j, indices in enumerate(cluster_indices):
                    # print('j = ', j, ', indices = ' + str(len(indices)))
                    inst_cluster = np.zeros((len(indices), 4), dtype=np.float32)
                    inst_cluster = sem_cluster[np.array(indices), 0:4]
                    inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), label_i, dtype=np.uint32)), axis=1)
                    # inst_cluster = np.insert(inst_cluster, 4, label_i, axis=1)
                    inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), inst_id, dtype=np.uint32)), axis=1)
                    inst_id = inst_id + 1
                    cluster.extend(inst_cluster) # Nx6

        # print(time.time()-start)
        # print('*'*80)
        cluster = np.array(cluster)
        if 'path' in FLAGS.pub_or_path:
            np.save(label_output_dir+'/'+label_name.split('/')[-1].split('.')[0]+".npy", cluster)
        if 'pub' in FLAGS.pub_or_path:
            # print(cluster[11100:11110])
            msg_points = pc2.create_cloud(header=self.header1, fields=_make_point_field(cluster.shape[1]), points=cluster)
            self._labels_pub.publish(msg_points)

        return cluster      

    def gen_graphs(self, FLAGS, scan_name, scan, graph_output_dir):
        inst = scan[:, -1] # get instance label
        inst_label_set = list(set(inst))  # get nums of inst
        inst_label_set.sort()
        # print("inst set: ", inst_label_set)
        nodes = []  # graph node
        edges = []  # graph edge
        weights = []  # graph edge weights
        cluster = []  # cluster -> node
        centers = []
        for id_i in range(len(inst_label_set)):
            index = np.argwhere(inst_label_set[id_i] == inst) # query cluster by instance label
            index = index.reshape(index.shape[0])
            inst_cluster = scan[index, :]
            sem_label = list(set(inst_cluster[:, -2])) # get semantic label
            assert len(sem_label) == 1 # one instance cluster should have only one semantic label
            if int(sem_label[0]) in node_map.keys():
                cluster.append(inst_cluster[:, :3])
                node_label = node_map[int(sem_label[0])]  # add node
                nodes.append(int(node_label))
                cluster_center = np.mean(inst_cluster[:, :3], axis=0)
                centers.append((cluster_center.tolist()))
            elif int(sem_label[0]) == 9 or int(sem_label[0]) == 10: # ignore "road" and "parking"
                continue
            else:
                print("wrong semantic label: ", sem_label[0])
                exit(-1)

        dist_thresh = 5 # less than thresh, add an edge between nodes

        for i in range(len(cluster)-1):
            for j in range(i+1, len(cluster)):
                pc_i = cluster[i]
                pc_j = cluster[j]
                center_i = np.mean(pc_i, axis=0)
                center_j = np.mean(pc_j, axis=0)
                center = np.mean([center_i, center_j], axis=0)  # centroid of the cluster

                index1 = np.argmin(np.linalg.norm(center - pc_i[:,None], axis=-1), axis=0)
                index2 = np.argmin(np.linalg.norm(center - pc_j[:,None], axis=-1), axis=0)
                min_dis = np.linalg.norm(pc_i[index1] - pc_j[index2], axis=-1)

                if min_dis <= dist_thresh:
                    edges.append([i, j])  # add edge
                    weight = float(1-min_dis/dist_thresh) #  w = 1 - d/d_thresh [0~5m] -> [1~0]
                    weights.append(weight) # add edge_weight
                else:
                    pass

        # generate graph
        graph = {"nodes": nodes,
                "edges": edges,
                "weights": weights,
                "centers": centers
                }

        # print(graph)
        if 'path' in FLAGS.pub_or_path:
            file_name = os.path.join(graph_output_dir, scan_name.split('/')[-1].split('.')[0]+".json")
            # print("output json: ", file_name)
            with open(file_name, "w", encoding="utf-8") as file: json.dump(graph, file)
        if 'pub' in FLAGS.pub_or_path:
            centers = np.array(centers)
            nodes = np.array(nodes)
            pub_nodes = np.concatenate((centers, nodes.reshape(-1, 1).astype(np.uint32)), axis=1)
            msg_points = pc2.create_cloud(header=self.header2, fields=_make_point_field(pub_nodes.shape[1]), points=pub_nodes)
            self._graph_pub.publish(msg_points)
            # rospy.loginfo(scan_names[frame])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./gen_graph.py")
    parser.add_argument('--dataset', '-d', type=str, required=False, default="/media/yxm/文档/data/kitti/dataset/", help='Dataset to calculate content. No Default')
    parser.add_argument('--config', '-c', type=str, required=False, default="config/semantic-kitti.yaml", help='Dataset config file. Defaults to %(default)s')
    parser.add_argument('--output_label', type=str, required=False, default="/media/kx/Semantic_KITTI/debug/labels", help='Output path for labels')
    parser.add_argument('--output_graph', type=str, required=False, default="/media/kx/Semantic_KITTI/debug/graphs", help='Output path for labels')
    parser.add_argument('--pub_or_path', type=str, required=False, default="path", help='pub_or_path')
    parser.add_argument('--pub_rate', type=int, default=10, help='the frequency(hz) of pc published, default `10`')
    parser.add_argument('--label_topic', type=str, default='/labeled_pc', help='the 3D point cloud message topic to be published, default `/labeled_pc`')
    parser.add_argument('--graph_topic', type=str, default='/graphs', help='the semantic graph message topic to be published, default `/graphs`')
    parser.add_argument('--demolition', type=bool, default=False, help='demolition or not')

    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Dataset", FLAGS.dataset)
    print("Config", FLAGS.config)
    print("*" * 80)

    # open config file
    try:
        print("Opening config file %s" % FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    # get training sequences to calculate statistics
    sequences = CFG["split"]["train"][:]
    color_map = CFG['color_map']
    learning_map_inv = CFG['learning_map_inv']
    print("Analizing sequences", sequences)

    # itearate over sequences
    for seq in sequences[:]:
        # make seq string
        print("*" * 80)
        seqstr = '{0:02d}'.format(int(seq))
        print("parsing seq {}".format(seq))

        # prepare output dir
        label_output_dir = FLAGS.output_label + '/' + seqstr
        if not os.path.exists(label_output_dir):
            os.makedirs(label_output_dir)

        graph_output_dir = FLAGS.output_graph + '/' + seqstr
        if not os.path.exists(graph_output_dir):
            os.makedirs(graph_output_dir)
        
        # does sequence folder exist?
        scan_paths = os.path.join(FLAGS.dataset, "sequences", seqstr, "velodyne")
        if os.path.isdir(scan_paths):
            print("Sequence folder exists!")
        else:
            print("Sequence folder doesn't exist! Exiting...")
            quit()
        # populate the pointclouds
        scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_paths)) for f in fn]
        scan_names.sort()

        # does sequence folder exist?
        label_paths = os.path.join(FLAGS.dataset, "sequences", seqstr, "labels")
        if os.path.isdir(label_paths):
            print("Labels folder exists!")
        else:
            print("Labels folder doesn't exist! Exiting...")
            quit()
        # populate the pointclouds
        label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_paths)) for f in fn]
        label_names.sort()

        # check that there are same amount of labels and scans
        # print(len(label_names))
        # print(len(scan_names))
        assert(len(label_names) == len(scan_names))

        # create a scan
        node = Semantic_kitti_node(FLAGS.pub_rate, FLAGS.label_topic, FLAGS.graph_topic)
        rate = rospy.Rate(FLAGS.pub_rate)
        for frame in tqdm(range(len(scan_names))):
            if rospy.is_shutdown():
                break
            cluster = node.gen_labels(FLAGS, scan_names[frame], label_names[frame], label_output_dir)
            node.gen_graphs(FLAGS, scan_names[frame], cluster, graph_output_dir)
            
            rate.sleep()
            # rospy.logwarn("%d frames published.", frame)