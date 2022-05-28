import numpy as np
import open3d as o3d

import mayavi.mlab as mlab
from utils import visualize_utils as V

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors


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

def test_plot3():
    x = np.arange(100)
    # Here are many sets of y to plot vs. x
    ys = x[:50, np.newaxis] + x[np.newaxis, :]

    segs = np.zeros((50, 100, 2))
    segs[:, :, 1] = ys
    segs[:, :, 0] = x
    # Mask some values to test masked array support:
    segs = np.ma.masked_where((segs > 50) & (segs < 60), segs)

    # We need to set the plot limits.
    fig, ax = plt.subplots()
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(ys.min(), ys.max())

    # *colors* is sequence of rgba tuples.
    # *linestyle* is a string or dash tuple. Legal string values are
    # solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq) where
    # onoffseq is an even length tuple of on and off ink in points.  If linestyle
    # is omitted, 'solid' is used.
    # See `matplotlib.collections.LineCollection` for more information.
    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

    line_segments = LineCollection(segs, linewidths=(0.5, 1, 1.5, 2),
                                   colors=colors, linestyle='solid')
    ax.add_collection(line_segments)
    ax.set_title('Line collection with masked arrays')
    plt.show()

if __name__ == "__main__":
    # test_plot_2([(1,0,0)])
    test_plot3()
    pass

