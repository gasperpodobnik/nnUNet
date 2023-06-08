#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.training.data_augmentation.data_augmentation_moreDA import (
    get_moreDA_augmentation,
)
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet_Dou import Generic_UNet_Dou
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import (
    default_2D_augmentation_params,
    get_patch_size,
    default_3D_augmentation_params,
)
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNet_variants.data_augmentation.nnUNetTrainerV2_noMirroringAxis2redRot import nnUNetTrainerV2_noMirroringAxis2redRot
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *

import torch

class KD_cost(nn.Module):
    def __init__(self, n_class):
        self.n_class = n_class
        
    def forward(self, source_logits, source_gt, target_logits):
        target_gt = source_gt
        kd_loss = 0.0

        self.source_logits = source_logits
        self.target_logits = target_logits

        self.source_prob = []
        self.target_prob = []

        temperature = 2.0
        eps = 1e-6
        
        # source_gt: batch_size, 1, patch_size, patch_size, patch_size

        for i in range(self.n_class):

            self.s_mask = (source_gt==i).repeat(1, self.n_class, 1, 1, 1)
            self.s_logits_mask_out = self.source_logits * self.s_mask
            self.s_logits_avg = torch.sum(self.s_logits_mask_out, dim=[0, 2, 3, 4]) / (torch.sum(source_gt==i) + eps)
            self.s_soft_prob = torch.nn.functional.softmax(self.s_logits_avg / temperature, dim=-1)

            self.source_prob.append(self.s_soft_prob)

            self.t_mask = (target_gt==i).repeat(1, self.n_class, 1, 1, 1)
            self.t_logits_mask_out = self.target_logits * self.t_mask
            self.t_logits_avg = torch.sum(self.t_logits_mask_out, dim=[0, 2, 3, 4]) / (torch.sum(target_gt==i) + eps)
            self.t_soft_prob = torch.nn.functional.softmax(self.t_logits_avg / temperature, dim=-1)

            self.target_prob.append(self.t_soft_prob)

            ## KL divergence loss
            loss = (torch.sum(self.s_soft_prob * torch.log(self.s_soft_prob / self.t_soft_prob)) +
                    torch.sum(self.t_soft_prob * torch.log(self.t_soft_prob / self.s_soft_prob))) / 2.0
            
            

            ## L2 Norm
            # loss = torch.nn.functional.mse_loss(self.s_soft_prob, self.t_soft_prob) / self.n_class

            kd_loss += loss

        return kd_loss / self.n_class


class nnUNetTrainerV2_DouKD(nnUNetTrainerV2_noMirroringAxis2redRot):
    # def process_plans(self, plans):
    #     super().process_plans(plans)
    #     self.batch_size = 1
        
    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array(
                [True]
                + [
                    True if i < net_numpool - 1 else False
                    for i in range(1, net_numpool)
                ]
            )
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            self.loss_KD = KD_cost(self.num_classes)
            ################# END ###################

            self.folder_with_preprocessed_data = join(
                self.dataset_directory,
                self.plans["data_identifier"] + "_stage%d" % self.stage,
            )
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!"
                    )

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr,
                    self.dl_val,
                    self.data_aug_params["patch_size_for_spatialtransform"],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                )
                self.print_to_log_file(
                    "TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                    also_print_to_console=False,
                )
                self.print_to_log_file(
                    "VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                    also_print_to_console=False,
                )
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file(
                "self.was_initialized is True, not running self.initialize again"
            )
        self.was_initialized = True
    
    def initialize_network(self):
        # importatante
        self.num_input_channels = 1
        
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {"eps": 1e-5, "affine": True}
        dropout_op_kwargs = {"p": 0, "inplace": True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        self.network = Generic_UNet_Dou(
            self.num_input_channels,
            self.base_num_features,
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage,
            2,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            net_nonlin,
            net_nonlin_kwargs,
            True,
            False,
            lambda x: x,
            InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes,
            self.net_conv_kernel_sizes,
            False,
            True,
            True
        )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
    
    
    def predict_preprocessed_data_return_seg_and_softmax(
        self,
        data: np.ndarray,
        do_mirroring: bool = True,
        mirror_axes: Tuple[int] = None,
        use_sliding_window: bool = True,
        step_size: float = 0.5,
        use_gaussian: bool = True,
        pad_border_mode: str = "constant",
        pad_kwargs: dict = None,
        all_in_gpu: bool = False,
        verbose: bool = True,
        mixed_precision=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(
            data,
            do_mirroring=do_mirroring,
            mirror_axes=mirror_axes,
            use_sliding_window=use_sliding_window,
            step_size=step_size,
            use_gaussian=use_gaussian,
            pad_border_mode=pad_border_mode,
            pad_kwargs=pad_kwargs,
            all_in_gpu=all_in_gpu,
            verbose=verbose,
            mixed_precision=mixed_precision,
        )
        self.network.do_ds = ds
        return ret

    def run_iteration(
        self, data_generator, do_backprop=True, run_online_evaluation=False
    ):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict["data"]
        target = data_dict["target"]
        
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        if self.fp16:
            self.optimizer.zero_grad()
            with autocast():
                output_0 = self.network(data[:,:1], True)
                output_1 = self.network(data[:,1:], False)
                del data
                l0 = self.loss(output_0, target) + self.loss(output_1, target)
                # [0] because other elements are the deep supervision logits
                l_kd = self.loss_KD.forward(output_0[0], target[0], output_1[0])
                l = l0 + l_kd

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            self.optimizer.zero_grad()
            output_0 = self.network(data[:,:1], True)
            output_1 = self.network(data[:,1:], False)
            del data
            l = self.loss(output_0, target) + self.loss(output_1, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
                
        if run_online_evaluation:
            self.run_online_evaluation(output_0, target)

        del target

        return l.detach().cpu().numpy()