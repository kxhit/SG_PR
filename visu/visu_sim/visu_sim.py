import pykitti
import numpy as np
import os
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

def _make_point_field(num_field, with_ring=True):
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


sim_dir = "/media/work/3D/SimGNN/kx/SG_LC/visu/visu_sim"
p_thresh = "5_20"
sq = "08"
dataset = pykitti.odometry('/media/data/kitti/odometry/dataset/', sq)

sq_poses = np.array(dataset.poses)
sq_xyz_i = sq_poses[:, :, -1]

# sq_sim_path = os.path.join(sim_dir, p_thresh,sq, sq+"_DL_db.npy")
sq_sim_path = os.path.join(sim_dir, p_thresh,sq, sq+"_M2DP_db.npy")
sq_similarity = np.load(sq_sim_path)
sq_xyz_i[:, -1] = sq_similarity

sim_pub = rospy.Publisher('sim', PointCloud2, queue_size=1)#1
fix_pub = rospy.Publisher('fix', PointCloud2, queue_size=1)#1
# ros node init
rospy.init_node('bin_node', anonymous=True)
rospy.loginfo("bin_node started.")

header = Header()
header.stamp = rospy.Time()
header.frame_id = "velodyne"

rate = rospy.Rate(10)

fix_id = 714   # todo

x = sq_xyz_i[:, 0].reshape(-1)
y = sq_xyz_i[:, 1].reshape(-1)
z = sq_xyz_i[:, 2].reshape(-1)
sim = sq_xyz_i[:, 3].reshape(-1)

sim = sim - np.min(sim) # normalize
sim = sim/np.max(sim)

# sim = np.exp(sim*50)
# sim = sim/np.max(sim)
print(sim)
cloud = np.stack((x, y, z, sim))

fix_x = sq_xyz_i[fix_id,0].reshape(-1)
fix_y = sq_xyz_i[fix_id,1].reshape(-1)
fix_z = sq_xyz_i[fix_id,2].reshape(-1)
fix_sim = sq_xyz_i[fix_id,3].reshape(-1)

fix_cloud = np.stack((fix_x,fix_y,fix_z, fix_sim))

while(1):
    msg_points = pc2.create_cloud(header=header, fields=_make_point_field(cloud.shape[0], with_ring=False), points=cloud.T)
    sim_pub.publish(msg_points)

    msg_points = pc2.create_cloud(header=header, fields=_make_point_field(fix_cloud.shape[0], with_ring=False),
                                  points=fix_cloud.T)
    fix_pub.publish(msg_points)

    rate.sleep()

# # tuli
# sim = np.arange(0,100,0.1)
# x = np.arange(0,100,0.1)
# y = np.arange(0,100,0.1)
# z = np.zeros((1000))
# cloud = np.stack((x, y, z, sim))
# while(1):
#     msg_points = pc2.create_cloud(header=header, fields=_make_point_field(cloud.shape[0], with_ring=False), points=cloud.T)
#     sim_pub.publish(msg_points)
#
#     rate.sleep()

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(sq_xyz_i[:, :3])
# rgb = np.tile(sim.reshape(-1,1),(1,3))
# rgb[:,1] = 0
# rgb[:,2] = 0
# print(rgb.shape)
# pcd.colors = o3d.utility.Vector3dVector(rgb)
#
# o3d.visualization.draw_geometries([pcd])