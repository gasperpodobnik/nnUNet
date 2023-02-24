import logging
logging.info('start\n\n\n')
import argparse
import json
import os
import pickle
import shutil
import sys
import copy
from os.path import isdir, join
from pathlib import Path
from tqdm import tqdm
import subprocess
import numpy as np
import pandas as pd
import nnunet.results.predict as predict
from pathlib import Path
from typing import Union

import SimpleITK as sitk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
import os
from os.path import join

from nnunet.training.model_restore import load_model_and_checkpoint_files
import torch
from torch.cuda.amp import autocast

sys.path.append(r"/media/medical/gasperp/projects")
sys.path.append(r"/media/medical/gasperp/projects/surface-distance")
from surface_distance import compute_metrics_deepmind



class nnUNet_Activation_Plotter_Arg_Parser(predict.nnUNet_Prediction_Arg_Parser):
    def __init__(self) -> None:
        super().__init__()
        self.parser.add_argument(
            "--slice_num",
            type=int,
            default=None,
            required=True,
            help="Patch slice index",
        )
        self.parser.add_argument(
            "--ct_patch",
            type=str,
            default='/media/medical/projects/head_and_neck/nnUnet/Task208_ONKOI-bothM-curatedFinal-MR-denoise/ct_patch_sitk43.nii.gz',
            help="",
        )
        self.parser.add_argument(
            "--mr_patch",
            type=str,
            default='/media/medical/projects/head_and_neck/nnUnet/Task208_ONKOI-bothM-curatedFinal-MR-denoise/mr_patch_sitk43.nii.gz',
            help="",
        )
        
    def __call__(self):
        # running in terminal
        self.args = vars(self.parser.parse_args())
        
        predict = nnUNet_Activation_Plotter(args=self.args)
        predict.execute_activaiton_plotting()
        
        
class nnUNet_Activation_Plotter(predict.Custom_nnUNet_Predict):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.slice_num = args['slice_num']
        self.results_dir_name = "activations"
        self.get_out_dir()
        self._setup_logging(log_dir=self.out_dir, clear_log=True)
        self.activations_out_dir = join(self.out_dir, self.config_str)
        
        self.ct_img = sitk.GetArrayFromImage(sitk.ReadImage(args['ct_patch']))
        self.mr_img = sitk.GetArrayFromImage(sitk.ReadImage(args['mr_patch']))
        
        if 'separate_encoders' in args['trainer_class_name']:
            self.hook_points = [
                ['conv_blocks_context', 0, 0, 'blocks', 1, 'conv'], 
                ['conv_blocks_context', 1, 0, 'blocks', 1, 'conv'], 
                ['conv_blocks_context', 0, 1, 'blocks', 1, 'lrelu'],
                ['conv_blocks_context', 1, 1, 'blocks', 1, 'lrelu'],
                ['conv_blocks_context', 0, 2, 'blocks', 0, 'conv'],
                ['conv_blocks_context', 1, 2, 'blocks', 0, 'conv'],
                ['conv_blocks_context', 0, 2, 'blocks', 0, 'lrelu'],
                ['conv_blocks_context', 1, 2, 'blocks', 0, 'lrelu'],
                ['conv_blocks_localization', 2, 1, 'blocks', 0, 'lrelu'],
                ['conv_blocks_localization', 3, 0, 'blocks', 0, 'conv'],
                ['conv_blocks_localization', 3, 1, 'blocks', 0, 'lrelu'],
                # ['tu', 0],
                ['tu', 1],
                ['tu', 2],
                ['tu', 3],
                ['tu', 4]
            ]
        else:
            self.hook_points = [
                ['conv_blocks_context', 0, 'blocks1', 0, 'conv'], 
                ['conv_blocks_context', 0, 'blocks2', 0, 'conv'], 
                ['conv_blocks_context', 0, 'blocks1', 1, 'conv'], 
                ['conv_blocks_context', 0, 'blocks2', 1, 'conv'], 
                # ['conv_blocks_context', 1, 'blocks1', 0, 'conv'],
                # ['conv_blocks_context', 1, 'blocks2', 0, 'conv'],
                # ['conv_blocks_context', 1, 'blocks1', 1, 'conv'],
                # ['conv_blocks_context', 1, 'blocks2', 1, 'conv'],
                ['conv_blocks_context', 1, 'blocks1', 1, 'lrelu'],
                ['conv_blocks_context', 1, 'blocks2', 1, 'lrelu'],
                ['conv_blocks_context', 2, 'blocks', 0, 'conv'],
                ['conv_blocks_context', 2, 'blocks', 0, 'lrelu'],
                ['conv_blocks_context', 2, 'blocks', 1, 'lrelu'],
                ['conv_blocks_context', 3, 'blocks', 0, 'lrelu'],
                ['conv_blocks_context', 3, 'blocks', 1, 'lrelu'],
                ['conv_blocks_localization', 2, 1, 'blocks', 0, 'lrelu'],
                ['conv_blocks_localization', 3, 0, 'blocks', 0, 'conv'],
                ['conv_blocks_localization', 3, 1, 'blocks', 0, 'lrelu'],
                # ['tu', 0],
                ['tu', 1],
                ['tu', 2],
                ['tu', 3],
                ['tu', 4]
            ]
        
    def generate_input_patches(self):
        self.options = [
            ['none', 1, 1],
            ['ct', 0, 1],
            ['mr', 1, 0],
            ]
        for blocked_modality, ct_f, mr_f in self.options:
            data = np.stack((self.ct_img*ct_f, self.mr_img*mr_f))[np.newaxis]
            data_torch = torch.from_numpy(data).float().to(device='cuda:0')
            yield blocked_modality, data_torch
        
    def execute_activaiton_plotting(self):
        self.trainer, self.all_params = self.load_model()
        self.state_dict = self.all_params[0]['state_dict']
        self.trainer.load_checkpoint_ram(self.all_params[0], False)
        
        self.network_activations = {}
        for blocked_modality, data_torch in self.generate_input_patches():
            actv_dict, self.all_layer_names = self.collect_activations(data_torch=data_torch)
            self.network_activations[blocked_modality] = copy.deepcopy(actv_dict)
            
        for layer_name in self.all_layer_names:
            activations = {bl_modal: actvs[layer_name] for bl_modal, actvs in self.network_activations.items()}
            vmin, vmax = get_percentiles(list(activations.values()))
            for blocked_modality, actv in activations.items():
                tmp = 40//actv.shape[2]
                plot_activations(activation=actv, actv_name=layer_name, vmin=vmin, vmax=vmax, blocked_modality=blocked_modality, slc=self.slice_num//tmp, img_base_dir=self.activations_out_dir)
            
        
            
        
    def print_layers(self):
        state_dict_keys = sorted(list(self.state_dict.keys()))
        for i in state_dict_keys:
            print(i)



    def collect_activations(self, data_torch):
        activations_dict = {}
        def get_activation(name):
            def hook(model, input, output):
                activations_dict[name] = output.detach().cpu().numpy()
            return hook
        
        self.trainer.network.to(device='cuda:0')
        self.trainer.network.eval()
        
        all_layer_names = []
        
        with autocast():
            with torch.no_grad():
                for hk in self.hook_points:
                    act_layer = get_attr_or_element(self.trainer.network, hk)
                    name = '_'.join([str(i) for i in hk])
                    act_layer.register_forward_hook(get_activation(name))
                    all_layer_names.append(name)
                output = self.trainer.network(data_torch)
        return activations_dict, all_layer_names
    
def get_attr_or_element(base, _list):
    for e, l in enumerate(_list):
        if isinstance(l, int):
            base = base[l]
        else:
            base = getattr(base, l)
        return get_attr_or_element(base, _list[e+1:])
    return base
    
        
def plot_activations(activation, actv_name, vmin, vmax, blocked_modality, slc, img_base_dir, show=False):
    os.makedirs(img_base_dir, exist_ok=True)
    num_filters = activation.shape[1]
    
    ncols_options = 2**np.arange(6)
    n_cols_and_rows = ncols_options[np.abs(ncols_options - np.sqrt(num_filters)).argmin()]
    n_cols_and_rows = [n_cols_and_rows, num_filters//n_cols_and_rows]
    ncols = max(n_cols_and_rows)
    nrows = min(n_cols_and_rows)
    
    fig_k = 1 if num_filters > 100 else 2
    figsize=(ncols*fig_k, nrows*fig_k)

    plt.close('all')
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for enum, ax in enumerate(axes.flatten()):
        ax.imshow(activation[0,enum,slc], cmap='gray', vmin=vmin, vmax=vmax)
        
    plt.suptitle(actv_name + ' ' + str(activation.shape) + f' DISPLAY vmin={vmin}, vmax={vmax}', fontsize=8)
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(join(img_base_dir, f'slc-{slc}_{actv_name}_BLOCKED-{blocked_modality}.png'))
    if not show:
        plt.close('all')
        
def get_percentiles(a):
    vmins = []
    vmaxs = []
    for ai in a:
        vmins.append(np.percentile(ai.flatten(), 1))
        vmaxs.append(np.percentile(ai.flatten(), 99))
    return np.median(vmins), np.median(vmaxs)

def plot_boxplot_activations(activation, actv_names, modalityblocked, img_base_dir, ylim=None):
    _list = [activation[a].flatten() for a in actv_names]
    ylimstr = ''
    if ylim is not None:
        _list = [np.clip(i, *ylim) for i in _list]
        ylimstr = 'clipped'
    plt.figure(figsize=(4, 10))
    plt.violinplot(_list)
    plt.xticks(np.arange(len(_list))+1, actv_names)
    plt.xticks(rotation=90, fontsize=7)
    plt.grid()
    
    
    plt.tight_layout()
    plt.show()
    plt.savefig(join(img_base_dir, f'{"--".join(actv_names)}_BLOCKED-{modalityblocked}_{ylimstr}.png'))
    
def main():
    run = nnUNet_Activation_Plotter_Arg_Parser()
    run()

if __name__ == "__main__":
    main()