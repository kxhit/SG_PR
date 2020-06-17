import numpy as np
import open3d as o3d
import os

base_dir = "/media/data/kitti/odometry/dataset/sequences"
sequence = "08"
bin_file = "000154.bin"
scan = np.fromfile(os.path.join(base_dir,sequence,"velodyne",bin_file),dtype=np.float32).reshape(-1,4)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(scan[:,:3])
o3d.visualization.draw_geometries_with_editing([pcd])
