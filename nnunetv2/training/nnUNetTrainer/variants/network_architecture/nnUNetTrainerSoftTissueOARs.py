import os
import torch
from torch import autocast
from typing import Union, Tuple, List

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainer_noMirroringAxis2redRot,
)
from nnunetv2.utilities.helpers import dummy_context
from torch.nn.parallel import DistributedDataParallel as DDP


class nnUNetTrainerSoftTissueOARs(nnUNetTrainer_noMirroringAxis2redRot):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
        num_epochs=1000,
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            unpack_dataset=unpack_dataset,
            device=device,
            num_epochs=num_epochs,
        )
        print("Using nnUNetTrainerSoftTissueOARs")
        self.adjust_targets()

    def adjust_targets(self):
        anchor_organs = [
            "Bone_Mandible",
            "Brainstem",
            "SpinalCord",
            "Eye_PR",
            "Eye_PL",
        ]
        new_target_organs = [
            "Glnd_Submand_L",
            "Glnd_Submand_R",
            "Parotid_L",
            "Parotid_R",
            "Pituitary",
            "Glnd_Lacrimal_L",
            "Glnd_Lacrimal_R",
        ]

        self.anchor_organs_dict = {
            a: self.label_manager.label_dict[a] for a in anchor_organs
        }

        self.new_target_organs_dict = {
            s: self.label_manager.label_dict[s] for s in new_target_organs
        }
        self.dataset_json["labels"] = {
            "background": 0,
            **self.new_target_organs_dict,
        }
        self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            raise NotImplementedError
        else:
            dl_tr = nnUNetDataLoader3D(
                dataset_tr,
                self.batch_size,
                initial_patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
            )
            dl_val = nnUNetDataLoader3D(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
            )
        return dl_tr, dl_val

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = (
                determine_num_input_channels(
                    self.plans_manager, self.configuration_manager, self.dataset_json
                )
                + 1
            )  # +1 for anchor mask

            self.network = self.build_network_architecture(
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                self.num_input_channels,
                enable_deep_supervision=True,
            ).to(self.device)
            # compile network for free speedup
            if ("nnUNet_compile" in os.environ.keys()) and (
                os.environ["nnUNet_compile"].lower() in ("true", "1", "t")
            ):
                self.print_to_log_file("Compiling network...")
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    def create_anchor_mask_and_prepare_target(self, data, old_target):
        anchor_mask = torch.zeros_like(old_target[0]).float()
        N_anchors = len(self.anchor_organs_dict)
        for enum, a in enumerate(self.anchor_organs_dict.values()):
            # add anchor organs to anchor mask
            # btw, it should be normalized to [0, 1]
            anchor_mask[old_target[0] == a] = (enum + 1) / N_anchors

        target = [torch.zeros_like(t) for t in old_target]
        for enum, s in enumerate(self.new_target_organs_dict.values()):
            for ot, t in zip(old_target, target):
                # add soft tissue organs to target
                t[ot == s] = enum + 1

        # add anchor mask to data
        data = torch.cat([data, anchor_mask], dim=1)

        return data, target

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        # importante!!!
        data, target = self.create_anchor_mask_and_prepare_target(data, target)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        with autocast(
            self.device.type, enabled=True
        ) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        # importante!!!
        data, target = self.create_anchor_mask_and_prepare_target(data, target)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(
            self.device.type, enabled=True
        ) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }


class nnUNetTrainerSoftTissueOARs_justMR(nnUNetTrainerSoftTissueOARs):
    def get_single_modality_from_batch(self, data: dict):
        return data[:, [1]]

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )  # +1 for anchor mask, -1 for CT, so default

            self.network = self.build_network_architecture(
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                self.num_input_channels,
                enable_deep_supervision=True,
            ).to(self.device)
            # compile network for free speedup
            if ("nnUNet_compile" in os.environ.keys()) and (
                os.environ["nnUNet_compile"].lower() in ("true", "1", "t")
            ):
                self.print_to_log_file("Compiling network...")
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    def train_step(self, batch: dict) -> dict:
        data = self.get_single_modality_from_batch(batch["data"])
        target = batch["target"]

        # importante!!!
        data, target = self.create_anchor_mask_and_prepare_target(data, target)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        with autocast(
            self.device.type, enabled=True
        ) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = self.get_single_modality_from_batch(batch["data"])
        target = batch["target"]

        # importante!!!
        data, target = self.create_anchor_mask_and_prepare_target(data, target)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(
            self.device.type, enabled=True
        ) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

class nnUNetTrainerSoftTissueOARs_justCT(nnUNetTrainerSoftTissueOARs_justMR):
    def get_single_modality_from_batch(self, data: dict):
        return data[:, [0]]


class nnUNetTrainerSmallOARs(nnUNetTrainerSoftTissueOARs):
    def adjust_targets(self):
        anchor_organs = [
            "Bone_Mandible",
            "Brainstem",
            "SpinalCord",
            "Eye_PR",
            "Eye_PL",
        ]
        new_target_organs = [
            "OpticNrv_L",
            "OpticNrv_R",
            "OpticChiasm",
            "Cochlea_L",
            "Cochlea_R",
            "Glnd_Lacrimal_L",
            "Glnd_Lacrimal_R",
            "Pituitary",
        ]

        self.anchor_organs_dict = {
            a: self.label_manager.label_dict[a] for a in anchor_organs
        }

        self.new_target_organs_dict = {
            s: self.label_manager.label_dict[s] for s in new_target_organs
        }
        self.dataset_json["labels"] = {
            "background": 0,
            **self.new_target_organs_dict,
        }
        self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)


class nnUNetTrainerSmallOARs_justMR(nnUNetTrainerSoftTissueOARs_justMR):
    def adjust_targets(self):
        anchor_organs = [
            "Bone_Mandible",
            "Brainstem",
            "SpinalCord",
            "Eye_PR",
            "Eye_PL",
        ]
        new_target_organs = [
            "OpticNrv_L",
            "OpticNrv_R",
            "OpticChiasm",
            "Cochlea_L",
            "Cochlea_R",
            "Glnd_Lacrimal_L",
            "Glnd_Lacrimal_R",
            "Pituitary",
        ]

        self.anchor_organs_dict = {
            a: self.label_manager.label_dict[a] for a in anchor_organs
        }

        self.new_target_organs_dict = {
            s: self.label_manager.label_dict[s] for s in new_target_organs
        }
        self.dataset_json["labels"] = {
            "background": 0,
            **self.new_target_organs_dict,
        }
        self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)


import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def get_bbox(
        self,
        data_shape: np.ndarray,
        force_fg: bool,
        class_locations: Union[dict, None],
        overwrite_class: Union[int, Tuple[int, ...]] = None,
        verbose: bool = False,
    ):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [
            data_shape[i]
            + need_to_pad[i] // 2
            + need_to_pad[i] % 2
            - self.patch_size[i]
            for i in range(dim)
        ]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    print("Warning! No annotated pixels in image!")
                    selected_class = None
                # print(f'I have ignore labels and want to pick a labeled area. annotated_classes_key: {self.annotated_classes_key}')
            elif force_fg:
                assert (
                    class_locations is not None
                ), "if force_fg is set class_locations cannot be None"
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), (
                        'desired class ("overwrite_class") does not '
                        "have class_locations (missing key)"
                    )
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [
                    i for i in class_locations.keys() if len(class_locations[i]) > 0
                ]

                # MODIFIED, previous behaviour was strange, we want to sample from classes that are in self.annotated_classes_key
                eligible_classes_or_regions = list(
                    set(self.annotated_classes_key) & set(eligible_classes_or_regions)
                )

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print("case does not contain any foreground classes")
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = (
                        eligible_classes_or_regions[
                            np.random.choice(len(eligible_classes_or_regions))
                        ]
                        if (
                            overwrite_class is None
                            or (overwrite_class not in eligible_classes_or_regions)
                        )
                        else overwrite_class
                    )
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError("lol what!?")
            voxels_of_that_class = (
                class_locations[selected_class] if selected_class is not None else None
            )

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[
                    np.random.choice(len(voxels_of_that_class))
                ]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [
                    max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2)
                    for i in range(dim)
                ]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(
                shape, force_fg, properties["class_locations"]
            )

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple(
                [slice(0, data.shape[0])]
                + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)]
            )
            data = data[this_slice]

            this_slice = tuple(
                [slice(0, seg.shape[0])]
                + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)]
            )
            seg = seg[this_slice]

            padding = [
                (-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0))
                for i in range(dim)
            ]
            data_all[j] = np.pad(
                data, ((0, 0), *padding), "constant", constant_values=0
            )
            seg_all[j] = np.pad(seg, ((0, 0), *padding), "constant", constant_values=-1)

        return {
            "data": data_all,
            "seg": seg_all,
            "properties": case_properties,
            "keys": selected_keys,
        }
