import numpy as np
import torch
import matplotlib
# switch off to enable visualization, switch on for training
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times New Roman']})
import open3d as o3d
# import mayavi.mlab as mlab
# from utils import visualize_utils as V
from matplotlib.animation import FuncAnimation
import sys
import os.path
from matplotlib import image
from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms, reconstructImg, reconstructPts, projectSceneFlow2Flow
import struct
from mpl_toolkits.mplot3d.axes3d import Axes3D
import torch.nn.functional as tf
from scipy import interpolate
import json
from losses import _adaptive_disocc_detection_disp, _adaptive_disocc_detection

def gif_data_scene_flow_2015(args):
    fig5, ax5 = plt.subplots()
    fig5.set_tight_layout(True)
    images_l_root = os.path.join(args.validation_dataset_root, "data_scene_flow", "training", "image_2_jpg")
    image_paths = []
    for ii in range(200):
        file_idx = '%.6d' % ii
        im_l1 = os.path.join(images_l_root, file_idx + "_10" + '.jpg')
        image_paths.append(im_l1)
        im_l2 = os.path.join(images_l_root, file_idx + "_11" + '.jpg')
        image_paths.append(im_l2)
    images = []
    for i in range(0, len(image_paths)):
        images.append(image.imread(image_paths[i]))

    def update(i):
        label = 'timestep {0}'.format(i)
        pic = ax5.imshow(images[i])
        ax5.set_xlabel(label)
        return pic, ax5

    anim = FuncAnimation(fig5, update, frames=np.arange(0, 201), interval=200)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('line.gif', dpi=80, writer='imagemagick')
    else:
        plt.show()

def gif_kitti_raw_dataset(args):
    fig6, ax6 = plt.subplots()
    fig6.set_tight_layout(True)
    path_dir = os.path.dirname(os.path.realpath(__file__))
    index_file = 'datasets/index_txt/kitti_full.txt'
    path_index_file = os.path.join(path_dir, index_file)
    if not os.path.exists(path_index_file):
        raise ValueError("Index File '%s' not found!", path_index_file)
    index_file = open(path_index_file, 'r')
    filenames = index_file.readlines()
    filenames = filenames[:int(len(filenames) / 100)]
    filename_list = [line.rstrip().split(' ') for line in filenames]
    image_paths = []
    images = []
    view1 = 'image_02/data'
    ext = '.jpg'
    for item in filename_list:
        date = item[0][:10]
        scene = item[0]
        idx_src = item[1]
        name_l1 = os.path.join(args.training_dataset_root, date, scene, view1, idx_src) + ext
        if os.path.isfile(name_l1):
            image_paths.append([name_l1])
    for i in range(0, len(image_paths)):
        images.append(image.imread(image_paths[i][0]))

    def update(i):
        label = 'timestep {0}'.format(i)
        pic = ax6.imshow(images[i])
        ax6.set_xlabel(label)
        return pic, ax6

    anim = FuncAnimation(fig6, update, frames=np.arange(0, len(image_paths)), interval=1)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('line.gif', dpi=80, writer='imagemagick')
    else:
        plt.show()


def gif_sceneflow_magnitude_and_components(figure_dict, data):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(25, 15));
    gs1 = gridspec.GridSpec(3, 2)
    gs1.update(wspace=0.025, hspace=0.05)
    fig.tight_layout()

    list_input_l1_aug = []
    list_input_l2_aug = []
    list_flow_f = []
    i = 0

    while True:
        try:
            list_input_l1_aug.append(data['input_l1_aug'+str(i)])
        except:
            break
        list_input_l2_aug.append(data['input_l2_aug' + str(i)])
        list_flow_f.append(data['flow_f' + str(i)])
        i =i+1

    ax[0, 0].set_title('left image at time t')
    ax[0, 0].imshow(list_input_l1_aug[0])
    ax[1, 0].set_title('left image at time t+1')
    ax[1, 0].imshow(list_input_l2_aug[0])
    ax[0, 1].imshow(list_input_l1_aug[0])
    pic = ax[0, 1].imshow(list_flow_f[0][:, :, 0], vmin=-0.9, vmax=0.9, cmap='jet', interpolation="none", alpha=0.8)
    ax[0, 1].set_title('x component of scene flow')
    ax[1, 1].imshow(list_input_l1_aug[0])
    pic = ax[1, 1].imshow(list_flow_f[0][:, :, 1],  vmin=-0.9, vmax=0.9, cmap='jet', interpolation="none", alpha=0.8)
    ax[1, 1].set_title('y component of scene flow')
    ax[2, 1].imshow(list_input_l1_aug[0])
    pic = ax[2, 1].imshow(list_flow_f[0][:, :, 2], vmin=-0.9, vmax=0.9, cmap='jet', interpolation="none", alpha=0.8)
    ax[2, 1].set_title('z component of scene flow')
    fig.colorbar(pic, ax=[ax[0,1], ax[1,1], ax[2,1]], shrink=0.6)
    # show magnitude of scene flow
    ax[2, 0].imshow(list_input_l1_aug[0])
    prod = torch.mul(list_flow_f[0], list_flow_f[0])
    sum = torch.sum(prod, dim=2)
    speed = torch.sqrt(sum).numpy()
    pic = ax[2, 0].imshow(speed, vmin=0, vmax=1.2, cmap='jet', interpolation="none", alpha=0.8) #vmin=0, vmax=1,
    ax[2, 0].set_title('magnitude of scene flow')
    fig.colorbar(pic, ax=ax[2, 0], shrink=0.6)

    def update(i):
        label = 'timestep {0}'.format(i)
        fig.suptitle(label)
        ax[0, 0].imshow(list_input_l1_aug[i])
        ax[1, 0].imshow(list_input_l2_aug[i])
        # plot x, y and z component of scene flow
        ax[0, 1].imshow(list_input_l1_aug[i])
        pic = ax[0, 1].imshow(list_flow_f[i][:, :, 0], vmin=-0.9, vmax=0.9, cmap='jet', interpolation="none", alpha=0.8)
        ax[1, 1].imshow(list_input_l1_aug[i])
        pic = ax[1, 1].imshow(list_flow_f[i][:, :, 1], vmin=-0.9, vmax=0.9, cmap='jet', interpolation="none", alpha=0.8)
        ax[2, 1].imshow(list_input_l1_aug[i])
        pic = ax[2, 1].imshow(list_flow_f[i][:, :, 2], vmin=-0.9, vmax=0.9, cmap='jet', interpolation="none", alpha=0.8)
        # show magnitude of scene flow
        ax[2, 0].imshow(list_input_l1_aug[i])
        prod = torch.mul(list_flow_f[i], list_flow_f[i])
        sum = torch.sum(prod, dim=2)
        speed = torch.sqrt(sum).numpy()
        pic = ax[2, 0].imshow(speed, vmin=0, vmax=1.2, cmap='jet', interpolation="none", alpha=0.8)  # vmin=0, vmax=1,
        return pic, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(list_flow_f)), interval=0)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('line.gif', dpi=80, writer='imagemagick')
    else:
        plt.show()

def reconstructed_image(figure_dict, key_prefix, path_prefix_saved_png, i, img_l2_warp, occ_map_f):
    fig=figure(figsize=(16, 12), dpi=80)
    fig.tight_layout()
    gs1 = gridspec.GridSpec(3, 1)
    gs1.update(wspace=0.025, hspace=5)
    ax1 = fig.add_subplot(311)
    ax1.imshow(img_l2_warp)
    ax1.set_title('reconstructed image at time t')
    occ_map_f_rep = occ_map_f.repeat(1, 1, 3)
    ax2 = fig.add_subplot(312)
    ax2.imshow(occ_map_f_rep.float())
    ax2.set_title('occlusion mask')
    img_l2_warp_occ = img_l2_warp.cpu() * occ_map_f_rep.float()
    ax3 = fig.add_subplot(313)
    ax3.set_title('reconstructed image at time t with applied occlusion mask')
    ax3.imshow(img_l2_warp_occ)
    figure_dict[key_prefix + 'reconstruced_image' + str(i)] = fig
    plt.savefig(path_prefix_saved_png+key_prefix + 'reconstruced_image' + str(i)+".png", dpi=300)
    return figure_dict

def ground_truth_after_augmentation__magnitude_and_components_of_scene_flow(figure_dict, key_prefix, path_prefix_saved_png, i, input_l1_aug, input_l2_aug, flow_f, data):
    fig=figure(figsize=(16,9), dpi=80) # 16,9
    fig.tight_layout()
    grid=fig.add_gridspec(nrows=3, ncols=2);
    plt.tight_layout()
    plt.rcParams.update({'font.size': 26,'text.usetex' : True})
    rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.subplots_adjust(wspace=0.01, hspace= 0.02, left=0.01, bottom=0.01, right=0.99, top=0.99) # wspace=0.01, hspace= 0.02, top=0.99, right=0.99, left=0.01,

    ax4 = fig.add_subplot(321)
    ax4.set_title(r'$\mathbf{I}_{t,l}$', fontname="Times New Roman")
    ax4.imshow(input_l1_aug)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax5 = fig.add_subplot(325)
    ax5.set_title('$\mathbf{I}_{t+1,l}$')
    im = ax5.imshow(input_l2_aug)
    ax5.set_xticks([])
    ax5.set_yticks([])



    """
    # plot direction and magnitude of scene flow
    torch.set_printoptions(threshold=1000000, linewidth=200)
    flow_f_lab = compute_color_sceneflow(flow_f.numpy())
    plt.figure(figsize=(25, 10))
    plt.imshow(flow_f_lab)
    """

    # plot x, y and z component of scene flow
    ax1 = fig.add_subplot(322)
    ax1.imshow(input_l1_aug)
    im = ax1.imshow(flow_f[:, :, 0], vmin=-1, vmax=1, cmap='jet', interpolation="none", alpha=0.8)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = fig.add_subplot(324)
    ax1.set_title('$sf^X_{fw,l}$')
    ax2.imshow(input_l1_aug)
    im = ax2.imshow(flow_f[:, :, 1],  vmin=-1, vmax=1, cmap='jet', interpolation="none", alpha=0.8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('$sf^Y_{fw,l}$')
    ax3 = fig.add_subplot(326)
    ax3.imshow(input_l1_aug)
    im = ax3.imshow(flow_f[:, :, 2], vmin=-1, vmax=1, cmap='jet', interpolation="none", alpha=0.8)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title('$sf^Z_{fw,l}$')
    fig.colorbar(im, ax=[ax1, ax2, ax3], shrink=0.91)


    # show magnitude of scene flow
    ax = fig.add_subplot(323)
    ax.imshow(input_l1_aug)
    prod = torch.mul(flow_f, flow_f)
    sum = torch.sum(prod, dim=2)
    speed = torch.sqrt(sum).numpy()
    im = ax.imshow(speed, vmin=0, vmax=1.2, cmap='jet', interpolation="none", alpha=0.8) #vmin=0, vmax=1,
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('$|\mathbf{sf}_{fw,l}|$')
    fig.colorbar(im, ax=[ax,ax4,ax5], shrink=0.2, aspect=4.5)
    #fig.colorbar(im, ax=ax, shrink=0.69, aspect=4)



    figure_dict[key_prefix + 'ground_truth_after_augmentation__magnitude_and_components_of_scene_flow' + str(i)] = fig
    plt.savefig(path_prefix_saved_png+key_prefix + 'ground_truth_after_augmentation__magnitude_and_components_of_scene_flow' + str(i)+".png", dpi=300)

    data['input_l1_aug'+str(i)] = input_l1_aug
    data['input_l2_aug'+str(i)] = input_l2_aug
    data['flow_f'+str(i)] = flow_f
    return figure_dict, data

def magnitude_and_components_of_scene_flow(figure_dict, key_prefix, path_prefix_saved_png, i, input_l1_aug, flow_f, data, present=True):
    fig=figure(figsize=(6,9), dpi=80) # 16,9
    fig.tight_layout()
    grid=fig.add_gridspec(nrows=4, ncols=1);
    plt.tight_layout()
    plt.rcParams.update({'font.size': 22,'text.usetex' : True}) #26
    rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.subplots_adjust(wspace=0.01, hspace= 0.5, left=0.01, bottom=0.03, right=0.90, top=0.95) # wspace=0.01, hspace= 0.02, top=0.99, right=0.99, left=0.01,

    # plot x, y and z component of scene flow
    ax1 = fig.add_subplot(412)
    ax1.imshow(input_l1_aug)
    im = ax1.imshow(flow_f[:, :, 0], vmin=-1, vmax=1, cmap='jet', interpolation="none", alpha=0.8)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = fig.add_subplot(413)
    ax1.set_title('$sf^X_{fw,l,pres}$')
    ax2.imshow(input_l1_aug)
    im = ax2.imshow(flow_f[:, :, 1],  vmin=-1, vmax=1, cmap='jet', interpolation="none", alpha=0.8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('$sf^Y_{fw,l,pres}$')
    ax3 = fig.add_subplot(414)
    ax3.imshow(input_l1_aug)
    im = ax3.imshow(flow_f[:, :, 2], vmin=-1, vmax=1, cmap='jet', interpolation="none", alpha=0.8)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title('$sf^Z_{fw,l,pres}$')
    fig.colorbar(im, ax=[ax1, ax2, ax3], shrink=0.98)#, shrink=0.91)


    # show magnitude of scene flow
    ax = fig.add_subplot(411)
    ax.imshow(input_l1_aug)
    prod = torch.mul(flow_f, flow_f)
    sum = torch.sum(prod, dim=2)
    speed = torch.sqrt(sum).numpy()
    im = ax.imshow(speed, vmin=0, vmax=1.2, cmap='jet', interpolation="none", alpha=0.8) #vmin=0, vmax=1,
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('$|\mathbf{sf}_{fw,l,pres}|$')
    fig.colorbar(im, ax=[ax], shrink=0.90, aspect=4.5)#, shrink=0.2, aspect=4.5)

    if present:
        ax1.set_title('$sf^X_{fw,l,pres}$')
        ax2.set_title('$sf^Y_{fw,l,pres}$')
        ax3.set_title('$sf^Z_{fw,l,pres}$')
        ax.set_title('$|\mathbf{sf}_{fw,l,pres}|$')
    else:
        ax1.set_title('$sf^X_{fw,l,fut}$')
        ax2.set_title('$sf^Y_{fw,l,fut}$')
        ax3.set_title('$sf^Z_{fw,l,fut}$')
        ax.set_title('$|\mathbf{sf}_{fw,l,fut}|$')

    figure_dict[key_prefix + 'ground_truth_after_augmentation__magnitude_and_components_of_scene_flow' + str(i)] = fig
    plt.savefig(path_prefix_saved_png+key_prefix + 'ground_truth_after_augmentation__magnitude_and_components_of_scene_flow' + str(i)+".png", dpi=300)

    data['input_l1_aug'+str(i)] = input_l1_aug
    # data['input_l2_aug'+str(i)] = input_l2_aug
    data['flow_f'+str(i)] = flow_f
    return figure_dict, data

def scene_flow_x(figure_dict, key_prefix, path_prefix_saved_png, i, input_l1_aug, flow_f, data, present=True):
    fig=figure(figsize=(20,9), dpi=80) # 16,9
    fig.tight_layout()
    grid=fig.add_gridspec(nrows=1, ncols=1);
    plt.tight_layout()
    plt.rcParams.update({'font.size': 22,'text.usetex' : True}) #26
    rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.subplots_adjust(wspace=0.01, hspace= 0.5, left=0.01, bottom=0.03, right=0.90, top=0.95) # wspace=0.01, hspace= 0.02, top=0.99, right=0.99, left=0.01,

    # plot x, y and z component of scene flow
    ax1 = fig.add_subplot(111)
    ax1.imshow(input_l1_aug)
    im = ax1.imshow(flow_f[:, :, 0], vmin=-0.3, vmax=0.3, cmap='jet', interpolation="none", alpha=0.4)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('$sf^Z_{fw,l,pres}$')
    fig.colorbar(im, ax=[ax1], shrink=0.54, aspect=4.5)

    if present:
        ax1.set_title('$sf^X_{fw,l,pres}$')

    figure_dict[key_prefix + 'ground_truth_after_augmentation__magnitude_and_components_of_scene_flow' + str(i)] = fig
    plt.savefig(path_prefix_saved_png+key_prefix + 'ground_truth_after_augmentation__magnitude_and_components_of_scene_flow' + str(i)+".png", dpi=300)

    data['input_l1_aug'+str(i)] = input_l1_aug
    # data['input_l2_aug'+str(i)] = input_l2_aug
    data['flow_f'+str(i)] = flow_f
    return figure_dict, data

def show_occlusion_map(figure_dict, key_prefix, path_prefix_saved_png, i, occ_map_f, original_input_image):
    occ_map_f_rep = occ_map_f.repeat(1, 1, 3)
    masked_original_input_image = original_input_image * occ_map_f_rep.float()

    fig=figure(figsize=(16, 12), dpi=80)
    fig.tight_layout()
    grid=fig.add_gridspec(nrows=1, ncols=1);
    grid.update(wspace=0.025, hspace=0.05)
    ax = fig.add_subplot(111)
    ax.set_title('left image at time t with mask')
    ax.imshow(masked_original_input_image)

    figure_dict[key_prefix + 'scene_flow_with_occlusion_mask' + str(i)] = fig
    plt.savefig(path_prefix_saved_png+key_prefix + 'masked_input_image' + str(i)+".png", dpi=300)
    return figure_dict

def direction_of_scene_flow(figure_dict, key_prefix, counter, flow_f):
    flow_f_vec = flow_f.numpy()
    # flow_f_vec = np.divide(flow_f_vec, speed[:, :, np.newaxis])
    reduce_size = 8
    flow_f_mean = np.zeros((np.int(256 / reduce_size), np.int(832 / reduce_size), 3))
    for i in range(0, np.int(832 / reduce_size)):
        for j in range(0, np.int(256 / reduce_size)):
            for k in range(0, 3):
                flow_f_mean[j, i, k] = np.mean(flow_f_vec[j:j + reduce_size, i:i + reduce_size, k], axis=(0, 1))

    x, y = np.meshgrid(np.arange(0, np.int(832 / reduce_size)), np.arange(0, np.int(256 / reduce_size)))
    fig, ax2 = plt.subplots(figsize=(25, 10));
    # ax2.xaxis.set_ticks([])
    # ax2.yaxis.set_ticks([])
    # ax2.axis([0, np.int(832/reduce_size), 0, np.int(256/reduce_size)])
    # ax2.set_aspect('equal')
    ax2.set_title('direction of scene flow')
    im = ax2.quiver(x, y, flow_f_mean[:, :, 0], flow_f_mean[:, :, 1], flow_f_mean[:, :, 2], scale=1e2, width=1e-3,
                    headwidth=3)
    fig.colorbar(im)
    figure_dict[key_prefix + 'direction_of_scene_flow' + str(counter)] = fig
    return figure_dict

def direction_of_scene_flow2(figure_dict, key_prefix, path_prefix_saved_png, counter, flow_f, input_l1_aug):
    fig= plt.figure();
    fig.tight_layout()
    ax = fig.add_subplot((111), projection='3d')
    resample_x = 4
    resample_y = 8
    flow_f = torch.div(flow_f, 10).numpy()
    flow_f = flow_f[::resample_x, ::resample_y, :]

    test=flow_f.shape[0]
    x, y = np.meshgrid(np.arange(0, flow_f.shape[1]*4 *resample_x, 4 * resample_x), np.arange(0, flow_f.shape[0]*4 *resample_y, 4 *resample_y))
    z = np.zeros((flow_f.shape[0], flow_f.shape[1]))
    ax.set_title('direction of scene flow')
    im = ax.quiver(x, y, z, flow_f[:, :, 0], flow_f[:, :, 1], flow_f[:, :, 2])
    ax.scatter(x, y, z, c='r', s=0.5)
    ax.view_init(elev=90, azim=-90)


    figure_dict[key_prefix + 'direction_of_scene_flow' + str(counter)] = fig
    plt.savefig(path_prefix_saved_png+ key_prefix + 'direction_of_scene_flow' + str(counter)+".png", dpi=300)
    return figure_dict

def voxelize(point_cloud1, point_cloud2, voxel_size, image):
    voxelized_point_cloud1 = []
    voxelized_point_cloud2 = []
    voxelized_image = []
    point_cloud1_mod = np.floor(point_cloud1 / voxel_size)
    min_x, max_x, min_y, max_y, min_z, max_z = -100, 100, -100, 100, 0, 100
    indices = (max_y - min_y) / voxel_size * (max_z - min_z) / voxel_size * (
            point_cloud1_mod[:, 0] - min_x / voxel_size) + (max_z - min_z) / voxel_size * (
                      point_cloud1_mod[:, 1] - min_y / voxel_size) + point_cloud1_mod[:, 2] - min_z / voxel_size
    point_cloud1 = np.concatenate((point_cloud1, indices[:, None]), axis=1)
    points = np.hstack((point_cloud1, point_cloud2, image))
    points = points[point_cloud1[:, 3].argsort()]
    point_cloud1=points[:,:4]
    point_cloud2 = points[:,4:7]
    image = points[:, 7:]
    i = 0
    while i < len(point_cloud1):
        same_voxel1 = [point_cloud1[i, :]]
        same_voxel2 = [point_cloud2[i, :]]
        same_voxel_im = [image[i,:]]
        j = i + 1
        while j < len(point_cloud1):
            if point_cloud1[i, 3] == point_cloud1[j, 3]:
                same_voxel1.append(point_cloud1[i, :])
                same_voxel2.append(point_cloud2[i, :])
                same_voxel_im.append(image[i,:])
                j += 1
            else:
                break
        voxelized_point_cloud1.append(np.mean(np.asarray(same_voxel1), axis=0))
        voxelized_point_cloud2.append(np.mean(np.asarray(same_voxel2), axis=0))
        voxelized_image.append(np.mean(np.asarray(same_voxel_im), axis=0))
        i = j
    voxelized_point_cloud1 = np.asarray(voxelized_point_cloud1)
    voxelized_point_cloud2 = np.asarray(voxelized_point_cloud2)
    voxelized_image = np.asarray(voxelized_image)
    return voxelized_point_cloud1[:,0:3], voxelized_point_cloud2, voxelized_image

def mask_points(computed_points, shifted_points, occ_map, image):
    computed_points_masked = computed_points
    shifted_points_masked = shifted_points
    occ_map = occ_map[:,:,None]
    occ_map = occ_map.view(-1,1).cpu().detach().numpy()
    for i in range(len(occ_map)-1,-1,-1):
        if not occ_map[i,0]:
            computed_points_masked = np.delete(computed_points_masked,i, axis=0)
            shifted_points_masked = np.delete(shifted_points_masked, i, axis=0)
            image = np.delete(image,i, axis=0)
            image_masked = image
    return computed_points_masked, shifted_points_masked, image_masked

def mask_points_fut(computed_points, occ_map, image):
    computed_points_masked = computed_points
    occ_map = occ_map[:,:,None]
    occ_map = occ_map.view(-1,1).cpu().detach().numpy()
    for i in range(len(occ_map)-1,-1,-1):
        if not occ_map[i,0]:
            computed_points_masked = np.delete(computed_points_masked,i, axis=0)
            image = np.delete(image,i, axis=0)
            image_masked = image
    return computed_points_masked, image_masked

def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    ctr = vis.get_view_control()
    ctr.rotate(5.8178 *180.0, 0.0, 0.0)

    ctr = vis.get_view_control()
    ctr.rotate(0.0, 5.8178 * 170.0, 0.0) # look from above
    ctr = vis.get_view_control()
    ctr.rotate(5.8178 * 180.0, 0.0, 0.0)
    vis.get_view_control().set_zoom(0.5)
    # screenshot over whole screen (workstation)
    # https://www.iloveimg.com/crop-image
    # crop image:
    # width=480
    # height=680
    # position x = 770
    # poxistiony = 400
    return False

def change_view(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    # ctr = vis.get_view_control()
    vis.get_view_control().set_zoom(0.15) #0.001
    vis.get_view_control().set_front([0.0, -0.35, -1.0])
    vis.get_view_control().set_lookat([0.0, -6.0, 6.0])
    vis.get_view_control().set_up([0.0, -100.0, 100.0])
    return False

angel=1
def rotate_x_plus(vis):
    opt = vis.get_render_option()
    ctr = vis.get_view_control()
    vis.get_view_control().rotate(5.8178 *angel, 0.0, 0.0)
    return False
def rotate_y_plus(vis):
    opt = vis.get_render_option()
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 5.8178 *angel, 0.0)
    return False
def rotate_z_plus(vis):
    opt = vis.get_render_option()
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 0.0, 5.8178 *angel)
    return False
def rotate_x_minus(vis):
    opt = vis.get_render_option()
    ctr = vis.get_view_control()
    ctr.rotate(-5.8178 *angel, 0.0, 0.0)
    return False
def rotate_y_minus(vis):
    opt = vis.get_render_option()
    ctr = vis.get_view_control()
    ctr.rotate(0.0, -5.8178 *angel, 0.0)
    return False
def rotate_z_minus(vis):
    opt = vis.get_render_option()
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 0.0, -5.8178 *angel)
    return False

def create_lines(point_cloud1, point_cloud2, step=1):
    points_ = np.vstack((point_cloud1[::step,:], point_cloud2[::step,:]))
    n = int(points_.shape[0]/2)
    lines = [[i, i + n] for i in range(n)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color((1, 0, 1))
    return line_set

def point_cloud_with_open3d(pts1, input_l1_aug, occ_map_f, pts1_tf=None, show_flow=True):


    # consider occlusion in point clouds of neural network
    computed_points1 = pts1.view(-1, 3).cpu().detach().numpy()

    input_l1_aug = input_l1_aug.view(-1, 3).cpu().detach().numpy()
    if show_flow:
        shifted_points1 = pts1_tf.view(-1, 3).cpu().detach().numpy()
        computed_points1_masked, shifted_points1_masked, input_l1_aug_masked = mask_points(computed_points1, shifted_points1, occ_map_f, input_l1_aug)
        computed_points1_masked_vox, shifted_points1_masked_vox, input_l1_aug_masked_vox = voxelize(computed_points1_masked, shifted_points1_masked, 0.5, input_l1_aug_masked)
    else:
        computed_points1_masked, input_l1_aug_masked = mask_points_fut(computed_points1, occ_map_f, input_l1_aug)

    if show_flow:
        shifted_points_after_masking = o3d.geometry.PointCloud()
        shifted_points_after_masking.points = o3d.utility.Vector3dVector(shifted_points1_masked[:,0:3])
        #shifted_points_after_masking.paint_uniform_color((1, 0, 0))
        shifted_points_after_masking.colors = o3d.utility.Vector3dVector(input_l1_aug)

        computed_points_after_masking_vox = o3d.geometry.PointCloud()
        computed_points_after_masking_vox.points = o3d.utility.Vector3dVector(computed_points1_masked_vox[:,0:3])
        computed_points_after_masking_vox.colors = o3d.utility.Vector3dVector(input_l1_aug_masked_vox)
        #computed_points_after_masking_vox.paint_uniform_color((0, 1, 0))

        shifted_points_after_masking_vox = o3d.geometry.PointCloud()
        shifted_points_after_masking_vox.points = o3d.utility.Vector3dVector(shifted_points1_masked_vox[:,0:3])
        shifted_points_after_masking_vox.paint_uniform_color((1, 0, 0))

        computed_points_after_masking = o3d.geometry.PointCloud()
        computed_points_after_masking.points = o3d.utility.Vector3dVector(computed_points1_masked[:, 0:3])
        computed_points_after_masking.colors = o3d.utility.Vector3dVector(input_l1_aug_masked)


    z_point = o3d.geometry.PointCloud()
    z_point.points = o3d.utility.Vector3dVector(np.array((0, 0, 0.11))[None, :])
    z_point.paint_uniform_color((0, 0, 0))

    computed_points_after_masking = o3d.geometry.PointCloud()
    computed_points_after_masking.points = o3d.utility.Vector3dVector(computed_points1_masked[:,0:3])
    computed_points_after_masking.colors = o3d.utility.Vector3dVector(input_l1_aug_masked)
    #computed_points_after_masking.paint_uniform_color((0, 1, 0))


    if show_flow:
        line_set = create_lines(computed_points1_masked_vox, shifted_points1_masked_vox, step=2)

    key_to_callback = {}
    key_to_callback[ord("K")] = change_view
    key_to_callback[ord("A")] = rotate_x_plus
    key_to_callback[ord("C")] = rotate_y_plus
    key_to_callback[ord("G")] = rotate_z_plus
    key_to_callback[ord("B")] = rotate_x_minus
    key_to_callback[ord("D")] = rotate_y_minus
    key_to_callback[ord("H")] = rotate_z_minus
    if show_flow:
        o3d.visualization.draw_geometries_with_key_callbacks([computed_points_after_masking, line_set], key_to_callback) # ([computed_points_after_masking, line_set], change_view)
    else:
        o3d.visualization.draw_geometries_with_key_callbacks([computed_points_after_masking], key_to_callback)





def point_cloud_with_mayavi(pts1, pts2):
    pts1_np = np.transpose(pts1[0].cpu().view(3, -1).data.numpy())
    pts2_np = np.transpose(pts2[0].cpu().view(3, -1).data.numpy())
    V.draw_scenes(points1=pts1_np, points2=pts2_np)
    mlab.show(stop=True)

def point_cloud_with_matplotlib(figure_dict, key_prefix, path_prefix_saved_png, counter, pts1, pts1_tf, input_l1_aug, input_l2_aug, occ_map_f):
    def zoom_factory(ax, base_scale=2.):
        def zoom_fun(event):
            # get the current x and y limits
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
            cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
            xdata = event.xdata  # get event x location
            ydata = event.ydata  # get event y location
            if event.button == 'up':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print
                event.button
            # set new limits
            ax.set_xlim([xdata - cur_xrange * scale_factor,
                         xdata + cur_xrange * scale_factor])
            ax.set_ylim([ydata - cur_yrange * scale_factor,
                         ydata + cur_yrange * scale_factor])
            plt.draw()  # force re-draw

        fig = ax.get_figure()
        # attach the call back
        # fig.canvas.mpl_connect('scroll_event', zoom_fun)
        # return the function
        figure_dict[key_prefix + 'point_cloud_with_matplotlib' + str(counter)] = fig
        return zoom_fun

    def plot_flow(points1, points2, path_prefix_saved_png, image):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        flow = points2 - points1

        x = points1[:, 0]
        y = points1[:, 1]
        z = points1[:, 2]
        u = flow[:, 0]
        v = flow[:, 1]
        w = flow[:, 2]

        ax.set_title('Point clouds with scene flow')
        ax.scatter(x, y, z, s=0.01, c=image, marker="s") # 'g'
        #ax.scatter(x + u, y + v, z + w, s=1, c='r')
        #ax.quiver(x, y, z, u, v, w, linewidths=0.3, length=1)
        #ax.legend(['Point Cloud at time t', 'Shifted point cloud', 'Scene flow'], bbox_to_anchor=(0.2, 1))
        ax.set_title('Point Cloud at time t')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_ylim([-15, 4])
        figure_dict[key_prefix + 'point_cloud_with_matplotlib_isometric'] = fig
        plt.savefig(path_prefix_saved_png+key_prefix + 'point_cloud_with_matplotlib_isometric' + ".png", dpi=300)

        ax.set_ylabel('')
        ax.view_init(elev=0, azim=-90)
        #ax.axes.yaxis.set_visible(False)
        ax.yaxis._axinfo['label']['space_factor'] = 2.0
        figure_dict[key_prefix + 'point_cloud_with_matplotlib_xz_plane'] = fig
        plt.savefig(path_prefix_saved_png+key_prefix + 'point_cloud_with_matplotlib_xz_plane' + ".png", dpi=300)

        ax.axes.yaxis.set_visible(True)
        ax.axes.zaxis.set_visible(True)
        ax.set_ylabel('Y axis')
        #plt.gca().axes.zaxis.set_ticklabels([])
        ax.set_zlabel('')
        ax.view_init(elev=-85, azim=-90)
        #ax.zaxis._axinfo['label']['space_factor'] = 2.0
        figure_dict[key_prefix + 'point_cloud_with_matplotlib_xy_plane'] = fig
        plt.savefig(path_prefix_saved_png+key_prefix + 'point_cloud_with_matplotlib_xy_plane' + ".png", dpi=300)

    # consider occlusion in point clouds of neural network
    computed_points1 = pts1.view(-1, 3).cpu().detach().numpy()
    shifted_points1 = pts1_tf.view(-1, 3).cpu().detach().numpy()
    input_l1_aug = input_l1_aug.view(-1, 3).cpu().detach().numpy()

    computed_points1_masked, shifted_points1_masked, input_l1_aug_masked = mask_points(computed_points1, shifted_points1, occ_map_f, input_l1_aug)
    computed_points1_masked_vox, shifted_points1_masked_vox, input_l1_aug_masked_vox = voxelize(computed_points1_masked, shifted_points1_masked, 0.05, input_l1_aug_masked)
    plot_flow(computed_points1_masked, shifted_points1_masked, path_prefix_saved_png, input_l1_aug_masked)
    return figure_dict

def disparity_and_depth(figure_dict, key_prefix, path_prefix_saved_png, i, input_l1_aug, input_l2_aug, disp_l1, disp_l2, depth1, depth2):
    fig, ax4 = plt.subplots(nrows=2, ncols=2, figsize=(14, 7));
    fig.tight_layout()
    # plot disparity
    ax4[0, 0].set_title('disparity of left image at time t')
    ax4[0, 0].imshow(input_l1_aug)
    im = ax4[0, 0].imshow(disp_l1, vmin=0, vmax=0.06, interpolation="none", alpha=0.8)
    ax4[1, 0].imshow(input_l1_aug)
    ax4[1, 0].set_title('disparity of left image at time t+1')
    im = ax4[1, 0].imshow(disp_l2, vmin=0, vmax=0.06, interpolation="none", alpha=0.8)
    fig.colorbar(im, ax=[ax4[0, 0], ax4[1, 0]], shrink=0.6)


    # plot depth
    ax4[0, 1].imshow(input_l1_aug)
    im = ax4[0, 1].imshow(depth1, vmin=0, vmax=80, interpolation="none", alpha=0.8)
    ax4[0, 1].set_title('depth of left image at time t')
    ax4[1, 1].imshow(input_l1_aug)
    im = ax4[1, 1].imshow(depth2, vmin=0, vmax=80, interpolation="none", alpha=0.8)
    ax4[1, 1].set_title('depth of left image at time t+1')
    fig.colorbar(im, ax=[ax4[0, 1], ax4[1, 1]], shrink=0.6)

    figure_dict[key_prefix + 'disparity_and_depth' + str(i)] = fig
    plt.savefig(path_prefix_saved_png+ key_prefix + 'disparity_and_depth' + str(i) + ".png", dpi=300)
    return figure_dict

def warping_layer_sf(x, sceneflow, disp, k1, input_size):

    _, _, h_x, w_x = x.size()
    disp = interpolate2d_as(disp, x) * w_x

    local_scale = torch.zeros_like(input_size)
    local_scale[:, 0] = h_x
    local_scale[:, 1] = w_x

    pts1, output_depth, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
    _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

    grid = coord1.transpose(1, 2).transpose(2, 3)
    x_warp = tf.grid_sample(x, grid, align_corners=True)

    mask = torch.ones_like(x, requires_grad=False)
    mask = tf.grid_sample(mask, grid, align_corners=True)
    mask = (mask >= 1.0).float()

    return x_warp * mask, pts1, pts1_tf, output_depth




def make_plots(figure_dict, show_train, show_eval, show_test,fig, all_inputs_val, all_outputs_val, counter, args, path_prefix_saved_png, all_inputs_test=None, all_outputs_test=None, all_inputs_train=None, all_outputs_train=None, original_input_image=None ):

    if show_train:
        all_inputs = all_inputs_train
        all_outputs = all_outputs_train
        key_prefix = 'train_'
        if not isinstance(all_inputs, list):
            all_inputs = [all_inputs]
            all_outputs = [all_outputs]
    elif show_eval:
        all_inputs = all_inputs_val
        all_outputs = all_outputs_val
        if not isinstance(all_inputs, list):
            all_inputs = [all_inputs]
            all_outputs = [all_outputs]
        key_prefix = 'val_'
        # gif_kitti_raw_dataset(args)
    elif show_test:
        all_inputs = all_inputs_test
        all_outputs = all_outputs_test
        if not isinstance(all_inputs, list):
            all_inputs = [all_inputs]
            all_outputs = [all_outputs]
        key_prefix = 'test_'
        # gif_data_scene_flow_2015(args)
    data = {}
    for i in range(0, len(all_inputs)):


        input_dict = all_inputs[i]
        output_dict = all_outputs[i]
        k1 = output_dict['k1']
        k2 = output_dict['k2']
        if args.version=="predict":
            input_l1_aug = torch.squeeze(input_dict['input_l2_aug'][0, :, :, :].cpu()).permute(1, 2, 0)
            input_l2_aug = torch.squeeze(input_dict['input_l3_aug'][0, :, :, :].cpu()).permute(1, 2, 0)
            flow_f_pred = output_dict['flowf_pred'][0]
            flow_b_pred = output_dict['flowb_pred'][0]
            disp_l2_pred = output_dict['displ2_pred'][0]
            disp_l3_pred = output_dict['displ3_pred'][0]
            x1_raw = output_dict['x2_raw']
            x2_raw = output_dict['x3_raw']
            k3 = output_dict['k3']
            flow_f = output_dict['flowf'][0]
            flow_b = output_dict['flowb'][0]
            disp_l1 = output_dict['displ1'][0]
            disp_l2 = output_dict['displ2'][0]
        else:
            x1_raw = output_dict['x1_raw']
            x2_raw = output_dict['x2_raw']
            flow_f = output_dict['flow_f'][0]
            flow_b = output_dict['flow_b'][0]
            disp_l1 = output_dict['disp_l1'][0]
            disp_l2 = output_dict['disp_l2'][0]
            input_l1_aug = torch.squeeze(input_dict['input_l1_aug'][0, :, :, :].cpu()).permute(1, 2, 0)
            input_l2_aug = torch.squeeze(input_dict['input_l2_aug'][0, :, :, :].cpu()).permute(1, 2, 0)

        _, pts1, pts1_tf, _ = warping_layer_sf(x1_raw, interpolate2d_as(flow_f, x1_raw),
                                                                interpolate2d_as(disp_l1, x1_raw), k1,
                                                                input_dict['aug_size'])

        if args.version=="predict":
            _, pts2, pts1_tf, _ = warping_layer_sf(x2_raw, interpolate2d_as(flow_f_pred, x2_raw),
                                                                    interpolate2d_as(disp_l2, x2_raw), k2,
                                                                    input_dict['aug_size'])

            _, _, pts3_tf, _ = warping_layer_sf(x2_raw, interpolate2d_as(flow_f_pred, x2_raw),
                                                                    interpolate2d_as(disp_l2, x2_raw), k2,
                                                                    input_dict['aug_size'])

        if args.version=="predict":
            x1_raw = torch.squeeze(x1_raw[0, :, :, :].cpu().permute(1, 2, 0))
            x2_raw = torch.squeeze(x2_raw[0, :, :, :].cpu().permute(1, 2, 0))
            flow_f_pred = torch.squeeze(flow_f_pred[0, :, :, :]).detach().cpu().permute(1, 2, 0)

        flow_f = torch.squeeze(flow_f[0, :, :, :]).detach().cpu().permute(1, 2, 0)
        flow_b = torch.squeeze(flow_b[0, :, :, :]).detach().cpu().permute(1, 2, 0)
        disp_l1 = torch.squeeze(disp_l1[0, :, :, :]).cpu().detach()
        disp_l2 = torch.squeeze(disp_l2[0, :, :, :]).cpu().detach()

        # depth1 = torch.squeeze(output_dict['depth_1'][0][0, :, :, :]).cpu().detach()
        # depth2 = torch.squeeze(output_dict['depth_2'][0][0, :, :, :]).cpu().detach()
        if show_train or show_eval:
            #  get occlusion map
            _, _, _, w_dp = output_dict['flow_f'][0].size()
            
            disp_occ_l1 = _adaptive_disocc_detection_disp(output_dict['output_dict_r']['disp_l1'][0]).detach()
            flow_b_proj = projectSceneFlow2Flow(output_dict['k1'], output_dict['flow_b'][0], output_dict['disp_l1'][0] * w_dp)
            occ_map_b_1 = _adaptive_disocc_detection(flow_b_proj).detach() * disp_occ_l1
            occ_map_b_1 = occ_map_b_1[0, :, :, :].cpu().detach().permute(1, 2, 0)
            flow_f_proj = projectSceneFlow2Flow(output_dict['k1'], output_dict['flow_f'][0], output_dict['disp_l1'][0] * w_dp)
            occ_map_f_1 = _adaptive_disocc_detection(flow_f_proj).detach() * disp_occ_l1
            occ_map_f_1 = occ_map_f_1[0, :, :, :].cpu().detach().permute(1, 2, 0)

            disp_occ_l2 = _adaptive_disocc_detection_disp(output_dict['output_dict_r']['disp_l2'][0]).detach()
            flow_f_proj = projectSceneFlow2Flow(output_dict['k2'], output_dict['flow_b'][0], output_dict['disp_l2'][0] * w_dp)
            occ_map_f_2 = _adaptive_disocc_detection(flow_f_proj).detach() * disp_occ_l2
            occ_map_f_2 = occ_map_f_2[0, :, :, :].cpu().detach().permute(1, 2, 0)
            flow_b_proj = projectSceneFlow2Flow(output_dict['k2'], output_dict['flow_b'][0], output_dict['disp_l2'][0] * w_dp)
            occ_map_b_2 = _adaptive_disocc_detection(flow_b_proj).detach() * disp_occ_l2
            occ_map_b_2 = occ_map_b_2[0, :, :, :].cpu().detach().permute(1, 2, 0)

            if args.version=="predict":
                disp_occ_l3_pred = _adaptive_disocc_detection_disp(output_dict['output_dict_r']['disp_l3_pred'][0]).detach()
                flow_b_pred = projectSceneFlow2Flow(output_dict['k2'], output_dict['flow_b_pred'][0], output_dict['disp_l3_pred'][0] * w_dp)
                occ_map_b3_pred = _adaptive_disocc_detection(flow_b_pred).detach() * disp_occ_l3_pred
                occ_map_b3_pred = occ_map_b3_pred[0, :, :, :].cpu().detach().permute(1, 2, 0)
                flow_f_pred_proj = projectSceneFlow2Flow(output_dict['k3'], output_dict['flow_f_pred'][0], output_dict['disp_l3_pred'][0] * w_dp)
                occ_map_f3_pred = _adaptive_disocc_detection(flow_f_pred_proj).detach() * disp_occ_l3_pred
                occ_map_f3_pred = occ_map_f3_pred[0, :, :, :].cpu().detach().permute(1, 2, 0)


            # get warped image
            # _, _, coord1 = pts2pixel_ms(output_dict['input_k_l1_aug'], output_dict['pts_1'][0], output_dict['flow_f'][0], [input_l1_aug.size(dim=0), input_l1_aug.size(dim=1)])
            # img_l2_warp = reconstructImg(coord1, input_dict['input_l2_aug'])
            # img_l2_warp = torch.squeeze(img_l2_warp[0, :, :, :].cpu()).detach().permute(1, 2, 0)



            pts1 = pts1[0, :, :, :].cpu().permute(1, 2, 0)
            pts1_tf = pts1_tf[0, :, :, :].cpu().permute(1, 2, 0)
            if args.version == "predict":
                pts2 = pts2[0, :, :, :].cpu().permute(1, 2, 0)
            # pts2_tf = pts2_tf[0, :, :, :].cpu().permute(1, 2, 0)
            # pts3_tf = pts3_tf[0, :, :, :].cpu().permute(1, 2, 0)

        else:
            # get mask
            gt_flow_mask = (input_dict['target_flow_mask'] == 1).float()
            gt_disp_mask = (input_dict['target_disp_mask'] == 1).float()
            gt_disp2_mask = (input_dict['target_disp2_mask_occ'] == 1).float()
            gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask
            # adjust size of mask to size of augmented image
            occ_map_f = interpolate2d_as(gt_sf_mask, input_dict['input_l1_aug'], mode="bilinear")
            occ_map_f = occ_map_f[0, :, :, :].cpu().detach().permute(1, 2, 0)
            # change format of original input image
            original_input_image = original_input_image[0, :, :, :].cpu().detach().permute(1, 2, 0)


        # if show_train or show_eval:
        #     figure_dict = reconstructed_image(figure_dict, key_prefix, path_prefix_saved_png, i, img_l2_warp, occ_map_f)
        # else:
        #     figure_dict = show_occlusion_map(figure_dict, key_prefix, path_prefix_saved_png, i, occ_map_f, input_l1_aug)



            # figure_dict = point_cloud_with_matplotlib(figure_dict, key_prefix, path_prefix_saved_png, i, pts1, pts1_tf,input_l1_aug, input_l2_aug, occ_map_f)
        if show_eval:
            # figure_dict, data = scene_flow_x(figure_dict, key_prefix, path_prefix_saved_png,i, input_l1_aug, tf.interpolate(flow_f[None, :, :, :].permute(0, 3, 1, 2), size=[128, 416], mode="bilinear", align_corners=True)[0,:, :, :].permute(1, 2, 0), data,present=True)

        # figure_dict, data = magnitude_and_components_of_scene_flow(figure_dict, key_prefix, path_prefix_saved_png, i, input_l1_aug, tf.interpolate(flow_f[None, :, :, :].permute(0, 3, 1, 2), size=[128, 416],mode="bilinear", align_corners=True)[0, :, :, :].permute(1, 2, 0), data, present=True)
        # figure_dict, data = magnitude_and_components_of_scene_flow(figure_dict, key_prefix, path_prefix_saved_png, i, input_l2_aug, tf.interpolate(flow_f_pred[None, :, :, :].permute(0, 3, 1, 2), size=[128, 416],mode="bilinear", align_corners=True)[0, :, :, :].permute(1, 2, 0), data, present=False)
        # plt.show()



            if args.version == "predict":
                pts1_flow_b = pts2 + tf.interpolate(flow_b[None, :, :, :].permute(0, 3, 1, 2), size=[128, 416],mode="bilinear", align_corners=True)[0, :, :, :].permute(1, 2, 0)
                pts2_flow_f_pred = pts2+tf.interpolate(flow_f_pred[None, :, :, :].permute(0, 3, 1, 2), size=[128, 416], mode="bilinear", align_corners=True)[0, :, :, :].permute(1, 2, 0)

            if args.version == "predict":
                point_cloud_with_open3d(pts1_flow_b, x2_raw, occ_map_f_1, show_flow=False)
                point_cloud_with_open3d(pts1_flow_b, x2_raw, occ_map_f_1, pts1_tf=pts2, show_flow=True)
                point_cloud_with_open3d(pts2, x2_raw, occ_map_f_1, show_flow=False)
                point_cloud_with_open3d(pts2, x2_raw, occ_map_f_1, pts1_tf=pts2_flow_f_pred, show_flow=True)
                point_cloud_with_open3d(pts2_flow_f_pred, x2_raw, occ_map_f_1, show_flow=False)
            # else:
            #     point_cloud_with_open3d(pts1, x2_raw, occ_map_f_1, show_flow=False)
            #     point_cloud_with_open3d(pts1, x2_raw, occ_map_f_1, pts1_tf=pts1_tf, show_flow=True)
            #     point_cloud_with_open3d(pts1_tf, x2_raw, occ_map_f_1, show_flow=False)



        # figure_dict = disparity_and_depth(figure_dict, key_prefix, path_prefix_saved_png, i, input_l1_aug, input_l2_aug, disp_l1, disp_l2, depth1, depth2)


        #point_cloud_with_mayavi(pts1, pts2)
        # figure_dict = direction_of_scene_flow2(figure_dict, key_prefix, path_prefix_saved_png, i, flow_f_no_upsample, input_l1_aug)


    # gif_sceneflow_magnitude_and_components(figure_dict, data)


    counter+=1
    return figure_dict


