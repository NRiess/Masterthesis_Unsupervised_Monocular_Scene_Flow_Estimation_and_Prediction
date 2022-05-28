import numpy as np
import open3d as o3d

#import mayavi.mlab as mlab
#from utils import visualize_utils as V


def test_plot_2(data):
    V.draw_scenes(
        points=np.array(data, dtype=int)
    )
    mlab.show(stop=True)


def test_plot(data):
    pcd1 = o3d.geometry.PointCloud()
    pts1_np = np.array(data, dtype=np.double)
    pcd1.points = o3d.utility.Vector3dVector(pts1_np)
    pcd1.paint_uniform_color([0, 0, 1])
    print(pcd1)
    o3d.visualization.draw_geometries([pcd1])

if __name__ == "__main__":
    test_plot([(1,0,0)])
    pass

