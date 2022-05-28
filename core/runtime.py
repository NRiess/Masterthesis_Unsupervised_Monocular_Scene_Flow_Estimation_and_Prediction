## Portions of Code from, copyright 2018 Jochen Gast

from __future__ import absolute_import, division, print_function

import numpy as np
import colorama
import logging
import collections
import torch
import torch.nn as nn

from core import logger, tools
from core.tools import MovingAverage

# for evaluation
from utils.flow import flow_to_png_middlebury, write_flow_png, write_depth_png
from create_plots import make_plots
from utils.sceneflow_util import compute_color_sceneflow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import io, color
from skimage.color import rgb2lab, lab2rgb
import os
# import open3d as o3d
#import mayavi.mlab as mlab
#from utils import visualize_utils as V
from datetime import datetime


# for tensorboardX
from torch.utils.tensorboard import SummaryWriter

# --------------------------------------------------------------------------------
# Exponential moving average smoothing factor for speed estimates
# Ranges from 0 (average speed) to 1 (current/instantaneous speed) [default: 0.3].
# --------------------------------------------------------------------------------
TQDM_SMOOTHING = 0


# -------------------------------------------------------------------------------------------
# Magic progressbar for inputs of type 'iterable'
# -------------------------------------------------------------------------------------------
def create_progressbar(iterable,
                       desc="",
                       train=False,
                       unit="it",
                       initial=0,
                       offset=0,
                       invert_iterations=False,
                       logging_on_update=False,
                       logging_on_close=True,
                       postfix=False):

    # ---------------------------------------------------------------
    # Pick colors
    # ---------------------------------------------------------------
    reset = colorama.Style.RESET_ALL
    bright = colorama.Style.BRIGHT
    cyan = colorama.Fore.CYAN
    dim = colorama.Style.DIM
    green = colorama.Fore.GREEN

    # ---------------------------------------------------------------
    # Specify progressbar layout:
    #   l_bar, bar, r_bar, n, n_fmt, total, total_fmt, percentage,
    #   rate, rate_fmt, rate_noinv, rate_noinv_fmt, rate_inv,
    #   rate_inv_fmt, elapsed, remaining, desc, postfix.
    # ---------------------------------------------------------------
    bar_format = ""
    bar_format += "%s==>%s%s {desc}:%s " % (cyan, reset, bright, reset)     # description
    bar_format += "{percentage:3.0f}%"                                      # percentage
    bar_format += "%s|{bar}|%s " % (dim, reset)                             # bar
    bar_format += " {n_fmt}/{total_fmt}  "                                  # i/n counter
    bar_format += "{elapsed}<{remaining}"                                   # eta
    if invert_iterations:
        bar_format += " {rate_inv_fmt}  "                                   # iteration timings
    else:
        bar_format += " {rate_noinv_fmt}  "
    bar_format += "%s{postfix}%s" % (green, reset)                          # postfix

    # ---------------------------------------------------------------
    # Specify TQDM arguments
    # ---------------------------------------------------------------
    tqdm_args = {
        "iterable": iterable,
        "desc": desc,                          # Prefix for the progress bar
        "total": len(iterable),                # The number of expected iterations
        "leave": True,                         # Leave progress bar when done
        "miniters": 1 if train else None,      # Minimum display update interval in iterations
        "unit": unit,                          # String be used to define the unit of each iteration
        "initial": initial,                    # The initial counter value.
        "dynamic_ncols": True,                 # Allow window resizes
        "smoothing": TQDM_SMOOTHING,           # Moving average smoothing factor for speed estimates
        "bar_format": bar_format,              # Specify a custom bar string formatting
        "position": offset,                    # Specify vertical line offset
        "ascii": True,
        "logging_on_update": logging_on_update,
        "logging_on_close": logging_on_close
    }

    return tools.tqdm_with_logging(**tqdm_args)


def tensor2float_dict(tensor_dict):
    return {key: tensor.item() for key, tensor in tensor_dict.items()}


def format_moving_averages_as_progress_dict(moving_averages_dict={},
                                            moving_averages_postfix="avg"):
    progress_dict = collections.OrderedDict([
        (key + moving_averages_postfix, "%1.4f" % moving_averages_dict[key].mean())
        for key in sorted(moving_averages_dict.keys())
    ])
    return progress_dict


def format_learning_rate(lr):
    if np.isscalar(lr):
        return "{}".format(lr)
    else:
        return "{}".format(str(lr[0]) if len(lr) == 1 else lr)


class TrainingEpoch:
    def __init__(self,
                 args,
                 loader,
                 augmentation=None,
                 tbwriter=None,
                 add_progress_stats={},
                 desc="Training Epoch"):

        self._args = args
        self._desc = desc
        self._loader = loader
        self._augmentation = augmentation
        self._add_progress_stats = add_progress_stats
        self._tbwriter = tbwriter

    def _step(self, example_dict, model_and_loss, optimizer, epoch):

        # Get input and target tensor keys
        input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
        target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
        tensor_keys = input_keys + target_keys

        # Possibly transfer to Cuda
        if self._args.cuda:
            for key, value in example_dict.items():
                if key in tensor_keys:
                    example_dict[key] = value.cuda(non_blocking=True)

        # Optionally perform augmentations
        if self._augmentation is not None:
            with torch.no_grad():
                example_dict = self._augmentation(example_dict)

        # Convert inputs/targets to variables that require gradients
        for key, tensor in example_dict.items():
            if key in input_keys:
                example_dict[key] = tensor.requires_grad_(True)
            elif key in target_keys:
                example_dict[key] = tensor.requires_grad_(False)

        # Reset gradients
        optimizer.zero_grad()

        # Run forward pass to get losses and outputs.
        loss_dict, loss_dict_tensorboard_train, output_dict, img_l2_warp, occ_map_f = model_and_loss(example_dict, epoch, self._args)

        # Check total_loss for NaNs
        training_loss = loss_dict[self._args.training_key]
        assert (not np.isnan(training_loss.item())), "training_loss is NaN"

        # Back propagation
        training_loss.backward()
        optimizer.step()

        # Return success flag, loss and output dictionary
        return loss_dict, loss_dict_tensorboard_train, output_dict, example_dict, img_l2_warp, occ_map_f

    def run(self, args, epoch, model_and_loss, optimizer):

        n_iter = 0
        model_and_loss.train()
        moving_averages_dict = None
        progressbar_args = {
            "iterable": self._loader,
            "desc": self._desc,
            "train": True,
            "offset": 0,
            "logging_on_update": False,
            "logging_on_close": True,
            "postfix": True
        }

        # Perform training steps
        with create_progressbar(**progressbar_args) as progress:
            all_inputs = []
            all_outputs = []
            counter = 0

            for example_dict in progress:

                # perform step
                loss_dict_per_step, loss_dict_tensorboard, output_dict, example_dict, img_l2_warp, occ_map_f = self._step(example_dict, model_and_loss, optimizer,epoch)
                all_inputs=[example_dict]
                all_outputs=[output_dict]
                # convert
                loss_dict_per_step = tensor2float_dict(loss_dict_per_step)

                # Possibly initialize moving averages and  Add moving averages
                if moving_averages_dict is None:
                    moving_averages_dict = {
                        key: MovingAverage() for key in loss_dict_per_step.keys()
                    }

                for key, loss in loss_dict_per_step.items():
                    moving_averages_dict[key].add_average(loss, addcount=self._args.batch_size)

                # view statistics in progress bar
                progress_stats = format_moving_averages_as_progress_dict(
                    moving_averages_dict=moving_averages_dict,
                    moving_averages_postfix="_ema")
                loss_dict_tensorboard = progress_stats

                progress.set_postfix(progress_stats)
                counter+=1
                if args.debug:
                    break

        # Return loss and output dictionary
        ema_loss_dict = { key: ma.mean() for key, ma in moving_averages_dict.items() }
        return ema_loss_dict, output_dict, all_inputs, all_outputs, loss_dict_tensorboard, img_l2_warp, occ_map_f


class EvaluationEpoch:
    def __init__(self,
                 args,
                 loader,
                 augmentation=None,
                 tbwriter=None,
                 add_progress_stats={},
                 desc="Evaluation Epoch"):
        self._args = args
        self._desc = desc
        self._loader = loader
        self._add_progress_stats = add_progress_stats
        self._augmentation = augmentation
        self._tbwriter = tbwriter
        self._save_output = False
        if self._args.save_flow or self._args.save_disp or self._args.save_disp2:
            self._save_output = True

    def save_outputs(self, example_dict, output_dict):

        # save occ
        save_root_flow = self._args.save + '/flow/'
        save_root_disp = self._args.save + '/disp_0/'
        save_root_disp2 = self._args.save + '/disp_1/'

        if self._args.save_flow:
            out_flow = output_dict["out_flow_pp"].data.cpu().numpy()
            b_size = output_dict["out_flow_pp"].data.size(0)
            file_names_flow = []
            for ii in range(0, b_size):
                file_name_flow = save_root_flow + '/' + str(example_dict["basename"][ii])
                file_names_flow.append(file_name_flow)
                directory_flow = os.path.dirname(file_name_flow)
                if not os.path.exists(directory_flow):
                    os.makedirs(directory_flow)
                    print(directory_flow, ' has been created.')

            for ii in range(0, b_size):
                # Vis
                flow_f_rgb = flow_to_png_middlebury(out_flow[ii, ...])
                file_name_flo_vis = file_names_flow[ii] + '_flow.png'
                io.imsave(file_name_flo_vis, flow_f_rgb)
                # Png output
                file_name = file_names_flow[ii] + '_10.png'
                write_flow_png(file_name, out_flow[ii, ...].swapaxes(0, 1).swapaxes(1, 2))

        if self._args.save_disp:

            b_size = output_dict["out_disp_l_pp"].data.size(0)
            out_disp = output_dict["out_disp_l_pp"].data.cpu().numpy()
            file_names_disp = []

            for ii in range(0, b_size):
                file_name_disp = save_root_disp + '/' + str(example_dict["basename"][ii])
                file_names_disp.append(file_name_disp)
                directory_disp = os.path.dirname(file_name_disp)
                if not os.path.exists(directory_disp):
                    os.makedirs(directory_disp)
                    print(directory_disp, ' has been created.')

            for ii in range(0, b_size):
                # Vis
                disp_ii = out_disp[ii, 0, ...]
                norm_disp = (disp_ii / disp_ii.max() * 255).astype(np.uint8)
                plt.imsave(file_names_disp[ii] + '_disp.jpg', norm_disp, cmap='plasma')
                # Png
                file_name = file_names_disp[ii] + '_10.png'
                write_depth_png(file_name, out_disp[ii, 0, ...])


        if self._args.save_disp2:

            b_size = output_dict["out_disp_l_pp_next"].data.size(0)
            out_disp2 = output_dict["out_disp_l_pp_next"].data.cpu().numpy()
            file_names_disp2 = []

            for ii in range(0, b_size):
                file_name_disp2 = save_root_disp2 + '/' + str(example_dict["basename"][ii])
                file_names_disp2.append(file_name_disp2)
                directory_disp2 = os.path.dirname(file_name_disp2)
                if not os.path.exists(directory_disp2):
                    os.makedirs(directory_disp2)
                    print(directory_disp2, ' has been created.')

            for ii in range(0, b_size):
                # Vis
                disp2_ii = out_disp2[ii, 0, ...]
                norm_disp2 = (disp2_ii / disp2_ii.max() * 255).astype(np.uint8)
                plt.imsave(file_names_disp2[ii] + '_disp2.jpg', norm_disp2, cmap='plasma')
                # Png
                file_name2 = file_names_disp2[ii] + '_10.png'
                write_depth_png(file_name2, out_disp2[ii, 0, ...])


    def _step(self, epoch, example_dict, model_and_loss):

        # Get input and target tensor keys
        input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
        target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
        tensor_keys = input_keys + target_keys

        # Possibly transfer to Cuda
        if self._args.cuda:
            for key, value in example_dict.items():
                if key in tensor_keys:
                    example_dict[key] = value.cuda(non_blocking=True)

        # Optionally perform augmentations
        if self._augmentation is not None:
            example_dict = self._augmentation(example_dict)

        # Run forward pass to get losses and outputs.
        loss_dict, loss_dict_tensorboard_val, output_dict, img_l2_warp, occ_map_f = model_and_loss(example_dict, epoch, self._args)

        return loss_dict, loss_dict_tensorboard_val, output_dict, example_dict, img_l2_warp, occ_map_f

    def run(self, args, epoch, model_and_loss):

        with torch.no_grad():

            # Tell model that we want to evaluate
            model_and_loss.eval()

            # Keep track of moving averages
            moving_averages_dict = None

            # Progress bar arguments
            progressbar_args = {
                "iterable": self._loader,
                "desc": self._desc,
                "train": False,
                "offset": 0,
                "logging_on_update": False,
                "logging_on_close": True,
                "postfix": True
            }

            # Perform evaluation steps
            with create_progressbar(**progressbar_args) as progress:
                all_inputs = []
                all_outputs = []
                counter = 0
                for example_dict in progress:

                    # Perform forward evaluation step
                    loss_dict_per_step, loss_dict_tensorboard_val, output_dict, example_dict, img_l2_warp, occ_map_f = self._step(epoch, example_dict, model_and_loss)
                    all_inputs = [example_dict]
                    all_outputs = [output_dict]

                    # Save results
                    if self._save_output:
                        self.save_outputs(example_dict, output_dict)

                    if isinstance(loss_dict_per_step, tuple):
                        loss_dict_per_step = loss_dict_per_step[0]
                    else:
                        loss_dict_per_step = loss_dict_per_step

                    # Convert loss dictionary to float
                    loss_dict_per_step = tensor2float_dict(loss_dict_per_step)

                    # Possibly initialize moving averages
                    if moving_averages_dict is None:
                        moving_averages_dict = {
                            key: MovingAverage() for key in loss_dict_per_step.keys()
                        }

                    # Add moving averages
                    for key, loss in loss_dict_per_step.items():
                        moving_averages_dict[key].add_average(loss, addcount=self._args.batch_size)

                    # view statistics in progress bar
                    progress_stats = format_moving_averages_as_progress_dict(
                        moving_averages_dict=moving_averages_dict,
                        moving_averages_postfix="")
                    loss_dict_tensorboard_val = progress_stats

                    progress.set_postfix(progress_stats)
                    counter+=1
                    if args.debug:
                        break


            # Record average losses
            avg_loss_dict = { key: ma.mean() for key, ma in moving_averages_dict.items() }
            
            return avg_loss_dict, output_dict, all_inputs, all_outputs, loss_dict_tensorboard_val, img_l2_warp, occ_map_f


class TestEpoch:
    def __init__(self,
                 args,
                 loader,
                 augmentation=None,
                 tbwriter=None,
                 add_progress_stats={},
                 desc="Test Epoch"):
        self._args = args
        self._desc = desc
        self._loader = loader
        self._add_progress_stats = add_progress_stats
        self._augmentation = augmentation
        self._tbwriter = tbwriter
        self._save_output = False
        if self._args.save_flow or self._args.save_disp or self._args.save_disp2:
            self._save_output = True

    def save_outputs(self, example_dict, output_dict):

        # save occ
        save_root_flow = self._args.save + '/flow/'
        save_root_disp = self._args.save + '/disp_0/'
        save_root_disp2 = self._args.save + '/disp_1/'

        if self._args.save_flow:
            out_flow = output_dict["out_flow_pp"].data.cpu().numpy()
            b_size = output_dict["out_flow_pp"].data.size(0)
            file_names_flow = []
            for ii in range(0, b_size):
                file_name_flow = save_root_flow + '/' + str(example_dict["basename"][ii])
                file_names_flow.append(file_name_flow)
                directory_flow = os.path.dirname(file_name_flow)
                if not os.path.exists(directory_flow):
                    os.makedirs(directory_flow)
                    print(directory_flow, ' has been created.')

            for ii in range(0, b_size):
                # Vis
                flow_f_rgb = flow_to_png_middlebury(out_flow[ii, ...])
                file_name_flo_vis = file_names_flow[ii] + '_flow.png'
                io.imsave(file_name_flo_vis, flow_f_rgb)
                # Png output
                file_name = file_names_flow[ii] + '_10.png'
                write_flow_png(file_name, out_flow[ii, ...].swapaxes(0, 1).swapaxes(1, 2))

        if self._args.save_disp:

            b_size = output_dict["out_disp_l_pp"].data.size(0)
            out_disp = output_dict["out_disp_l_pp"].data.cpu().numpy()
            file_names_disp = []

            for ii in range(0, b_size):
                file_name_disp = save_root_disp + '/' + str(example_dict["basename"][ii])
                file_names_disp.append(file_name_disp)
                directory_disp = os.path.dirname(file_name_disp)
                if not os.path.exists(directory_disp):
                    os.makedirs(directory_disp)
                    print(directory_disp, ' has been created.')

            for ii in range(0, b_size):
                # Vis
                disp_ii = out_disp[ii, 0, ...]
                norm_disp = (disp_ii / disp_ii.max() * 255).astype(np.uint8)
                plt.imsave(file_names_disp[ii] + '_disp.jpg', norm_disp, cmap='plasma')
                # Png
                file_name = file_names_disp[ii] + '_10.png'
                write_depth_png(file_name, out_disp[ii, 0, ...])

        if self._args.save_disp2:

            b_size = output_dict["out_disp_l_pp_next"].data.size(0)
            out_disp2 = output_dict["out_disp_l_pp_next"].data.cpu().numpy()
            file_names_disp2 = []

            for ii in range(0, b_size):
                file_name_disp2 = save_root_disp2 + '/' + str(example_dict["basename"][ii])
                file_names_disp2.append(file_name_disp2)
                directory_disp2 = os.path.dirname(file_name_disp2)
                if not os.path.exists(directory_disp2):
                    os.makedirs(directory_disp2)
                    print(directory_disp2, ' has been created.')

            for ii in range(0, b_size):
                # Vis
                disp2_ii = out_disp2[ii, 0, ...]
                norm_disp2 = (disp2_ii / disp2_ii.max() * 255).astype(np.uint8)
                plt.imsave(file_names_disp2[ii] + '_disp2.jpg', norm_disp2, cmap='plasma')
                # Png
                file_name2 = file_names_disp2[ii] + '_10.png'
                write_depth_png(file_name2, out_disp2[ii, 0, ...])

    def _step(self, epoch, example_dict, model_and_loss):

        # Get input and target tensor keys
        input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
        target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
        tensor_keys = input_keys + target_keys

        # Possibly transfer to Cuda
        if self._args.cuda:
            for key, value in example_dict.items():
                if key in tensor_keys:
                    example_dict[key] = value.cuda(non_blocking=True)

        # Optionally perform augmentations
        if self._augmentation is not None:
            example_dict = self._augmentation(example_dict)

        # Run forward pass to get losses and outputs.
        loss_dict, loss_dict_tensorboard_test, output_dict, occ_map_f = model_and_loss(example_dict, epoch, self._args)

        return loss_dict, loss_dict_tensorboard_test, output_dict, example_dict, occ_map_f

    def run(self, args, model_and_loss, epoch):

        with torch.no_grad():

            # Tell model that we want to evaluate
            model_and_loss.eval()

            # Keep track of moving averages
            moving_averages_dict = None

            # Progress bar arguments
            progressbar_args = {
                "iterable": self._loader,
                "desc": self._desc,
                "train": False,
                "offset": 0,
                "logging_on_update": False,
                "logging_on_close": True,
                "postfix": True
            }

            # Perform test steps
            with create_progressbar(**progressbar_args) as progress:
                all_inputs = []
                all_outputs = []
                counter=0
                for example_dict in progress:


                    # Perform forward evaluation step
                    loss_dict_per_step, loss_dict_tensorboard_test, output_dict, example_dict, occ_map_f= self._step(epoch, example_dict,
                                                                                                          model_and_loss)
                    all_inputs = example_dict
                    all_outputs = output_dict

                    # Save results
                    if self._save_output:
                        self.save_outputs(example_dict, output_dict)

                    if isinstance(loss_dict_per_step, tuple):
                        loss_dict_per_step = loss_dict_per_step[0]
                    else:
                        loss_dict_per_step = loss_dict_per_step

                    # Convert loss dictionary to float
                    loss_dict_per_step = tensor2float_dict(loss_dict_per_step)

                    # Possibly initialize moving averages
                    if moving_averages_dict is None:
                        moving_averages_dict = {
                            key: MovingAverage() for key in loss_dict_per_step.keys()
                        }

                    # Add moving averages
                    for key, loss in loss_dict_per_step.items():
                        moving_averages_dict[key].add_average(loss, addcount=self._args.batch_size)

                    # view statistics in progress bar
                    progress_stats = format_moving_averages_as_progress_dict(
                        moving_averages_dict=moving_averages_dict,
                        moving_averages_postfix="")
                    loss_dict_tensorboard_test = progress_stats
                    progress.set_postfix(progress_stats)

                    counter+=1
                    if args.debug:
                        break



            # Record average losses
            avg_loss_dict = {key: ma.mean() for key, ma in moving_averages_dict.items()}

            return avg_loss_dict, output_dict, all_inputs, all_outputs, loss_dict_tensorboard_test, occ_map_f



def exec_runtime(args,
                 checkpoint_saver,
                 model_and_loss,
                 optimizer,
                 lr_scheduler,
                 train_loader,
                 validation_loader,
                 test_loader,
                 training_augmentation,
                 validation_augmentation,
                 test_augmentation):

    # ----------------------------------------------------------------------------------------------
    # Tensorboard writer
    # ----------------------------------------------------------------------------------------------
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S ")
    tensorBoardWriter = SummaryWriter(args.save + '/writer '+ dt_string)
    path_to_tensorboard = args.save + '/writer '+ dt_string



    if train_loader is not None:
        training_module = TrainingEpoch(
            args, 
            desc="   Train",
            loader=train_loader,
            augmentation=training_augmentation,
            tbwriter=tensorBoardWriter)

    if validation_loader is not None:
        evaluation_module = EvaluationEpoch(
            args,
            desc="Validate",
            loader=validation_loader,
            augmentation=validation_augmentation,
            tbwriter=tensorBoardWriter)

    if test_loader is not None:
        test_module = TestEpoch(
            args,
            desc="   Test",
            loader=test_loader,
            augmentation=test_augmentation,
            tbwriter=tensorBoardWriter)

    # --------------------------------------------------------
    # Log some runtime info
    # --------------------------------------------------------
    with logger.LoggingBlock("Runtime", emph=True):
        logging.info("start_epoch: %i" % args.start_epoch)
        logging.info("total_epochs: %i" % args.total_epochs)

    # ---------------------------------------
    # Total progress bar arguments
    # ---------------------------------------
    progressbar_args = {
        "desc": "Progress",
        "initial": args.start_epoch - 1,
        "invert_iterations": True,
        "iterable": range(1, args.total_epochs + 1),
        "logging_on_close": True,
        "logging_on_update": True,
        "postfix": False,
        "unit": "ep"
    }

    # --------------------------------------------------------
    # Total progress bar
    # --------------------------------------------------------
    print(''), logging.logbook('')
    total_progress = create_progressbar(**progressbar_args)
    print("\n")

    # --------------------------------------------------------
    # Remember validation loss
    # --------------------------------------------------------
    best_validation_loss = float("inf") if args.validation_key_minimize else -float("inf")
    store_as_best = False

    args.start_epoch=1
    figure_dict = {}
    for epoch in range(args.start_epoch, args.total_epochs + 1):
        with logger.LoggingBlock("Epoch %i/%i" % (epoch, args.total_epochs), emph=True):

            # --------------------------------------------------------
            # Always report learning rate
            # --------------------------------------------------------
            if lr_scheduler is None:
                logging.info("lr: %s" % format_learning_rate(args.optimizer_lr))
            else:
                logging.info("lr: %s" % format_learning_rate(lr_scheduler.get_lr()))

            # -------------------------------------------
            # Create and run a training epoch
            # -------------------------------------------
            if train_loader is not None:
                avg_loss_dict, output_dict, all_inputs_train, all_outputs_train, loss_dict_tensorboard_train, img_l2_warp, occ_map_f = training_module.run(args,epoch, model_and_loss=model_and_loss, optimizer=optimizer)

            # -------------------------------------------
            # Run with test set
            # -------------------------------------------
            if not args.version=="predict":
                if test_loader is not None:
                    args.test = True
                    model_and_loss.test=True
                    avg_loss_dict, output_dict, all_inputs_test, all_outputs_test, loss_dict_tensorboard_test, occ_map_f_test = test_module.run(args,
                        model_and_loss=model_and_loss, epoch=epoch)


            model_and_loss.test=False
            args.test = False

            # -------------------------------------------
            # Create and run a validation epoch
            # -------------------------------------------
            if validation_loader is not None:

                # ---------------------------------------------------
                # Construct holistic recorder for epoch
                # ---------------------------------------------------
                args.validation = True
                model_and_loss.validation=True
                avg_loss_dict, output_dict, all_inputs_val, all_outputs_val,loss_dict_tensorboard_val, img_l2_warp_val, occ_map_f_val = evaluation_module.run(args, epoch, model_and_loss=model_and_loss)
                model_and_loss.validation=False
                args.validation=False


                counter = 0
                fig=[]
                # figure_dict = {}
                path_prefix_saved_png = path_to_tensorboard +"/epoch_"+str(epoch)+"_"
                show_train, show_eval, show_test = True, False, False
                if train_loader is not None:
                    figure_dict = make_plots(figure_dict, show_train, show_eval, show_test, fig, all_inputs_val, all_outputs_val, counter, args, path_prefix_saved_png, all_inputs_train=all_inputs_train, all_outputs_train=all_outputs_train)
                show_train, show_eval, show_test = False, True, False
                if validation_loader is not None:
                    figure_dict = make_plots(figure_dict, show_train, show_eval, show_test, fig, all_inputs_val, all_outputs_val, counter, args, path_prefix_saved_png)
                    show_train, show_eval, show_test = False, False, True
                if not args.version=="predict":
                    if test_loader is not None:
                        figure_dict = make_plots(figure_dict, show_train, show_eval, show_test, fig, all_inputs_val, all_outputs_val, counter, args, path_prefix_saved_png, all_inputs_test=all_inputs_test, all_outputs_test=all_outputs_test, original_input_image=all_inputs_test["input_l1"])


                # ----------------------------------------------------------------
                # Evaluate whether this is the best validation_loss
                # ----------------------------------------------------------------
                validation_loss = avg_loss_dict[args.validation_key]
                if args.validation_key_minimize:
                    store_as_best = validation_loss < best_validation_loss
                else:
                    store_as_best = validation_loss > best_validation_loss
                if store_as_best:
                    best_validation_loss = validation_loss

            # --------------------------------------------------------
            # Tensorboard X writing
            # --------------------------------------------------------
            if 'loss_dict_tensorboard_val' in locals():
                for key, value in loss_dict_tensorboard_val.items():
                    tensorBoardWriter.add_scalar('val/' + key, float(value), epoch)
            if 'loss_dict_tensorboard_train' in locals():
                for key, value in loss_dict_tensorboard_train.items():
                    tensorBoardWriter.add_scalar('train/' + key, float(value), epoch)
            if 'loss_dict_tensorboard_test' in locals():
                for key, value in loss_dict_tensorboard_test.items():
                    tensorBoardWriter.add_scalar('test/' + key, float(value), epoch)

            for key, value in figure_dict.items():
                tensorBoardWriter.add_figure(key+'/epoch_{}'.format(epoch), value, epoch, close=True)
            plt.close('all')

            # --------------------------------------------------------
            # Update standard learning scheduler
            # --------------------------------------------------------
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)


            # ----------------------------------------------------------------
            # Also show best loss on total_progress
            # ----------------------------------------------------------------
            total_progress_stats = {
                "best_" + args.validation_key + "_avg": "%1.4f" % best_validation_loss
            }
            total_progress.set_postfix(total_progress_stats)


            # # ----------------------------------------------------------------
            # # Bump total progress
            # # ----------------------------------------------------------------
            total_progress.update()
            print('')

            # ----------------------------------------------------------------
            # Store checkpoint
            # ----------------------------------------------------------------
            if checkpoint_saver is not None:
                checkpoint_saver.save_latest(
                    directory=args.save,
                    model_and_loss=model_and_loss,
                    stats_dict=dict(avg_loss_dict, epoch=epoch),
                    dt_string=dt_string,
                    store_as_best=store_as_best)

            # ----------------------------------------------------------------
            # Vertical space between epochs
            # ----------------------------------------------------------------
            print(''), logging.logbook('')

    # ----------------------------------------------------------------
    # Finish
    # ----------------------------------------------------------------
    if args.evaluation is False:
        tensorBoardWriter.close()

    total_progress.close()
    logging.info("Finished.")
