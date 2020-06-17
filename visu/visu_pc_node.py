"""
Visualize pointcloud using ros rviz
Convert .bin files to ros pointcloud2 data with rings
Nx6
xyzi(0-1) sem instance
"""

import argparse
import os
import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import json

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

def _make_point_field(num_field, with_ring=False):
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

    msg_pf4 = pc2.PointField()
    msg_pf4.name = np.str('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)#float64
    msg_pf4.count = np.uint32(1)

    if num_field == 4:
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]
    elif num_field == 5:
        if with_ring == True:
            msg_pf5 = pc2.PointField()
            msg_pf5.name = np.str('ring')
            msg_pf5.offset = np.uint32(20)
            msg_pf5.datatype = np.uint8(4)#4 int16
            msg_pf5.count = np.uint32(1)
            # print("here debug")
        else:
            msg_pf5 = pc2.PointField()
            msg_pf5.name = np.str('label')
            msg_pf5.offset = np.uint32(20)
            msg_pf5.datatype = np.uint8(7)
            msg_pf5.count = np.uint32(1)
            # msg_pf5 = pc2.PointField()
            # msg_pf5.name = np.str('label')
            # msg_pf5.offset = np.uint32(20)
            # msg_pf5.datatype = np.uint8(4)
            # msg_pf5.count = np.uint32(1)
    elif num_field ==6:
        msg_pf5 = pc2.PointField()
        msg_pf5.name = np.str('semantic')
        msg_pf5.offset = np.uint32(20)
        msg_pf5.datatype = np.uint8(7)  # 4 int16
        msg_pf5.count = np.uint32(1)

        msg_pf6 = pc2.PointField()
        msg_pf6.name = np.str('instance')
        msg_pf6.offset = np.uint32(24)
        msg_pf6.datatype = np.uint8(7)  # 4 int16
        msg_pf6.count = np.uint32(1)

        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5, msg_pf6]
    else:
        print("please check num_field: ", num_field)
        exit(-1)

    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]

def get_quadrant(point):
    res = 0
    x = point[0]
    y = point[1]
    if x > 0 and y >= 0:
        res = 1
    elif x <= 0 and y > 0:
        res = 2
    elif x < 0 and y <= 0:
        res = 3
    elif x >= 0 and y < 0:
        res = 4
    return res

def add_ring_info(scan_points):
    num_of_points = scan_points.shape[0]
    scan_points = np.hstack([scan_points,
                             np.zeros((num_of_points, 1),dtype=np.int8)])
    velodyne_rings_count = 64
    previous_quadrant = 0
    ring = 0
    for num in range(num_of_points - 1, -1, -1):
        quadrant = get_quadrant(scan_points[num])
        if quadrant == 4 and previous_quadrant == 1 and ring < velodyne_rings_count - 1:
            ring += 1
        scan_points[num, 4] = int(ring)
        previous_quadrant = quadrant
    return scan_points


def rotate_point_cloud_z(lidar_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nxf array, original batch of point clouds
        Return:
          Nxf array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(lidar_data.shape, dtype=np.float32)
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.zeros(lidar_data.shape)
    rotated_data = lidar_data
    rotated_data[:, :3] = np.dot(lidar_data[:, :3], rotation_matrix)
    return rotated_data

class KITTINode(object):
    """
    A ros node to publish velodyne pointcloud with rings
    """

    def __init__(self, dataset_path=' ',
                 pub_rate=10,
                 pub_velody_points_topic='/kitti/velodyne_points'):
        """
        ros node spin in init function

        :param dataset_path:
        :param pub_rate:
        :param pub_velody_points_topic:
        """

        self._path = dataset_path
        self._pub_rate = pub_rate
        # publisher
        self._velodyne_points_pub = rospy.Publisher(pub_velody_points_topic, PointCloud2, queue_size=1)#1
        pub_rotate_velody_points_topic = "/semantic_nodes"
        self._nodes_pub = rospy.Publisher(pub_rotate_velody_points_topic, PointCloud2, queue_size=1)  # 1
        # ros node init
        rospy.init_node('bin_node', anonymous=True)
        rospy.loginfo("bin_node started.")

        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = "velodyne"

        rate = rospy.Rate(self._pub_rate)
        cnt = 0

        # get npy files
        bin_files = []
        if os.path.isdir(self._path):
            for f in os.listdir(self._path):
                if os.path.isdir(f):
                    continue
                else:
                    bin_files.append(f)
            bin_files.sort()

        # publish pointcloud
        while(1):
            for f in bin_files[4:5]:
            # for f in bin_files:
                if rospy.is_shutdown():
                    break
    
                self.publish_pointcloud(self._path + "/" + f, header)
                cnt += 1
    
                rate.sleep()

        rospy.logwarn("%d frames published.", cnt)

    def publish_pointcloud(self, bin_file, header):
        # print(bin_file)
        graph_base_dir = "/media/work/data/kitti/odometry/semantic-kitti/graph/00"
        file_name = bin_file.strip('.npy').split('/')[-1]
        graph_json = os.path.join(graph_base_dir,file_name+".json")
        graph_data = json.load(open(graph_json))
        nodes_center = graph_data["centers"]
        nodes_sem = graph_data["nodes"]
        record = np.load(bin_file)
        # record = np.fromfile(bin_file, dtype=np.float32).reshape(-1,6) #(Nx6) xyzi sem ins
        lidar = record  # x, y, z, intensity
        lidar = rotate_point_cloud_z(lidar)
        # lidar_label = record[:,4]
        # lidar = add_ring_info(lidar) # ring 0-63(int)
        # add sem 12
        for i in range(1,20):
            fake_p = [0,0,0,0,i,-1]
            lidar = np.row_stack((lidar, fake_p))
        x = lidar[:, 0].reshape(-1)
        y = lidar[:, 1].reshape(-1)
        z = lidar[:, 2].reshape(-1)
        intensity = lidar[:, 3].reshape(-1) #* 255  # 0~1 -> 0~255
        # ring = lidar[:, 4].reshape(lidar.shape[0],1).astype(np.int16)
        # label = lidar_label.reshape(lidar.shape[0],1).astype(np.int16)
        sem = lidar[:, 4].reshape(-1).astype(np.int16)
        ins = lidar[:, 5].reshape(-1).astype(np.int16)
        for i in range(1,20):
            index = np.where(sem == i)[0].reshape(-1)
            if i in node_map.keys():
                sem[index] = node_map[i]
                # print(node_map[i])
            else:
                sem[index] = -1
        # print(np.max(sem))
        assert np.max(sem) <= 11
        # cloud = []
        cloud = np.stack((x, y, z, intensity, sem, ins))
        # cloud = np.stack((x, y, z, intensity, sem))

        msg_points = pc2.create_cloud(header=header, fields=_make_point_field(cloud.shape[0], with_ring=False), points=cloud.T)
        self._velodyne_points_pub.publish(msg_points)
        for i in range(12):
            nodes_center.append([0,0,1000])
            nodes_sem.append(i)
        nodes_center = np.array(nodes_center).reshape(-1,3) # nx3
        nodes_sem = np.array(nodes_sem).reshape(-1)
        node_x = nodes_center[:,0].reshape(-1)
        node_y = nodes_center[:,1].reshape(-1)
        node_z = nodes_center[:,2].reshape(-1)

        nodes = np.stack((node_x,node_y,node_z,nodes_sem))

        msg_nodes = pc2.create_cloud(header=header, fields=_make_point_field(nodes.shape[0], with_ring=False),
                                      points=nodes.T)
        self._nodes_pub.publish(msg_nodes)

        file_name = bin_file.strip('.npy').split('/')[-1]
        rospy.loginfo("%s published.", file_name)
        rospy.loginfo("cloud shape: %d.", lidar.shape[0])


if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(description='KITTI velodyne point cloud publisher')
    parser.add_argument('--dataset_path', type=str,
                        help='the path of training dataset, default `/media/work/data/kitti/odometry/semantic-kitti/sem_kitti_cluster/00',
                        default='/media/work/data/kitti/odometry/semantic-kitti/sem_kitti_cluster/00')
    parser.add_argument('--pub_rate', type=int,
                        help='the frequency(hz) of image published, default `10`',
                        default=10)
    parser.add_argument('--pub_velodyne_points_topic', type=str,
                        help='the 3D point cloud message topic to be published, default `/velodyne_points`',
                        default='/velodyne_points')
    args = parser.parse_args()

    # start npy2pintcloud node
    node = KITTINode(dataset_path=args.dataset_path,
                           pub_rate=args.pub_rate,
                           pub_velody_points_topic=args.pub_velodyne_points_topic)

    rospy.logwarn('finished.')