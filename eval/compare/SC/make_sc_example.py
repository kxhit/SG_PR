import os 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from open3d import *


def rotate_point_cloud(pc, axis="y", max_angle=10):
    """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    assert pc.shape[1] == 3
    rotation_angle = np.random.uniform() * 2 * np.pi * max_angle/360
    # rotation_angle = 1 * 2 * np.pi * max_angle/360

    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)

    if axis == "y":
        # along y pitch
        rotation_matrix = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])

    elif axis == "x":
        # along x roll
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cosval, -sinval],
                                    [0, sinval, cosval]])
    elif axis == "z":
        # along z yaw
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
    else:
        print("axis wrong: ", axis)
        exit(-1)
    rotated_data = np.dot(pc, rotation_matrix)
    return rotated_data


class kitti_vlp_database:
    def __init__(self, bin_dir):
        self.bin_dir = bin_dir
        self.bin_files = os.listdir(bin_dir); self.bin_files.sort()    
  
        self.num_bins = len(self.bin_files)
       
    
class ScanContext:
 
    # static variables 
    viz = 0
    
    downcell_size = 0.5
    
    kitti_lidar_height = 2.0;
    
    # sector_res = np.array([45, 90, 180, 360, 720])
    # ring_res = np.array([10, 20, 40, 80, 160])
    sector_res = np.array([60])
    ring_res = np.array([20])
    max_length = 80
    
     
    def __init__(self, bin_dir, bin_file_name, random_rotate=False, mode="default"):

        self.bin_dir = bin_dir
        self.bin_file_name = bin_file_name
        self.bin_path = bin_dir + bin_file_name
        self.random_rotate = random_rotate
        self.mode = mode
        self.scancontexts = self.genSCs()

        
    def load_velo_scan(self):
        if self.mode == "lln":
            if self.bin_path.split(".")[-1] == "bin":
                scan = np.fromfile(self.bin_path, dtype=np.float32)
                scan = scan.reshape((-1, 3))
                ptcloud_xyz = scan
        else:
            if self.bin_path.split(".")[-1] == "bin":
                scan = np.fromfile(self.bin_path, dtype=np.float32)
                scan = scan.reshape((-1, 4))
                ptcloud_xyz = scan[:, :-1]
            else:
                scan = np.load(self.bin_path)
                ptcloud_xyz = scan[:, :3]
        if self.random_rotate == True:
            ptcloud_xyz = rotate_point_cloud(ptcloud_xyz, "z", 360)
            # ptcloud_xyz = rotate_point_cloud(ptcloud_xyz, "x", 10)
            # ptcloud_xyz = rotate_point_cloud(ptcloud_xyz, "y", 10)
        elif self.random_rotate == "fov":
            yaw = np.arctan((ptcloud_xyz[:,1]/ptcloud_xyz[:,0]))/np.pi*180
            ind_1 = np.argwhere(yaw >= -50)
            ind_2 = np.argwhere(yaw <= 50)
            ind = np.array(list(set(list(ind_1.reshape(-1))).intersection(list(ind_2.reshape(-1)))))
            ptcloud_xyz = ptcloud_xyz[ind, :]
        print(ptcloud_xyz.shape)
        return ptcloud_xyz
        
        
    def xy2theta(self, x, y):
        if (x >= 0 and y >= 0): 
            theta = 180/np.pi * np.arctan(y/x);
        if (x < 0 and y >= 0): 
            theta = 180 - ((180/np.pi) * np.arctan(y/(-x)));
        if (x < 0 and y < 0): 
            theta = 180 + ((180/np.pi) * np.arctan(y/x));
        if ( x >= 0 and y < 0):
            theta = 360 - ((180/np.pi) * np.arctan((-y)/x));

        return theta
            
        
    def pt2rs(self, point, gap_ring, gap_sector, num_ring, num_sector):
        x = point[0]
        y = point[1]
        z = point[2]
        
        if(x == 0.0):
            x = 0.001
        if(y == 0.0):
            y = 0.001
     
        theta = self.xy2theta(x, y)
        faraway = np.sqrt(x*x + y*y)
        
        idx_ring = np.divmod(faraway, gap_ring)[0]       
        idx_sector = np.divmod(theta, gap_sector)[0]

        if(idx_ring >= num_ring):
            idx_ring = num_ring-1 # python starts with 0 and ends with N-1
        
        return int(idx_ring), int(idx_sector)
    
    
    def ptcloud2sc(self, ptcloud, num_sector, num_ring, max_length):
        
        num_points = ptcloud.shape[0]
       
        gap_ring = max_length/num_ring
        gap_sector = 360/num_sector
        
        enough_large = 1000
        sc_storage = np.zeros([enough_large, num_ring, num_sector])
        sc_counter = np.zeros([num_ring, num_sector])
        
        for pt_idx in range(num_points):

            point = ptcloud[pt_idx, :]
            point_height = point[2] + ScanContext.kitti_lidar_height
            
            idx_ring, idx_sector = self.pt2rs(point, gap_ring, gap_sector, num_ring, num_sector)
            
            if sc_counter[idx_ring, idx_sector] >= enough_large:
                continue
            sc_storage[int(sc_counter[idx_ring, idx_sector]), idx_ring, idx_sector] = point_height
            sc_counter[idx_ring, idx_sector] = sc_counter[idx_ring, idx_sector] + 1

        sc = np.amax(sc_storage, axis=0)
            
        return sc

    
    def genSCs(self):
        ptcloud_xyz = self.load_velo_scan()
        # print("The number of original points: " + str(ptcloud_xyz.shape) )
    
        pcd = geometry.PointCloud()
        pcd.points = utility.Vector3dVector(ptcloud_xyz)
        downpcd = geometry.PointCloud.voxel_down_sample(pcd, voxel_size = ScanContext.downcell_size)
        ptcloud_xyz_downed = np.asarray(downpcd.points)
        # print("The number of downsampled points: " + str(ptcloud_xyz_downed.shape) )
        # draw_geometries([downpcd])
    
        if(ScanContext.viz):
            visualization.draw_geometries([downpcd])
            
        self.SCs = []
        for res in range(len(ScanContext.sector_res)):
            num_sector = ScanContext.sector_res[res]
            num_ring = ScanContext.ring_res[res]
            
            sc = self.ptcloud2sc(ptcloud_xyz_downed, num_sector, num_ring, ScanContext.max_length)
            self.SCs.append(sc)

            
    def plot_multiple_sc(self, fig_idx=1):

        num_res = len(ScanContext.sector_res)

        fig, axes = plt.subplots(nrows=num_res)
     
        axes[0].set_title('Scan Contexts with multiple resolutions', fontsize=14)
        for ax, res in zip(axes, range(num_res)):
            ax.imshow(self.SCs[res])
            
        plt.show()

    
if __name__ == "__main__":

    bin_dir = './data/'
    bin_db = kitti_vlp_database(bin_dir)
    
    for bin_idx in range(bin_db.num_bins):
        
        bin_file_name = bin_db.bin_files[bin_idx]
        bin_path = bin_db.bin_dir + bin_file_name
        
        sc = ScanContext(bin_dir, bin_file_name)

        fig_idx = 1
        # sc.plot_multiple_sc(fig_idx)

        print(len(sc.SCs))
        print(sc.SCs[0].shape)

        
        
        
        
        
        
        
