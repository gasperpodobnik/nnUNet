import inspect
import multiprocessing
import os
from pathlib import Path
import traceback
import sys
import logging
from SegmentationEvalMetrics import compute_metrics_multilabel
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional
import SimpleITK as sitk
import numpy as np
import torch
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import (
    load_json,
    join,
    isfile,
    maybe_mkdir_p,
    isdir,
    subdirs,
    save_json,
)
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import pandas as pd
import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import (
    PreprocessAdapterFromNpy,
    preprocessing_iterator_fromfiles,
    preprocessing_iterator_fromnpy,
)
from nnunetv2.inference.export_prediction import (
    export_prediction_from_logits,
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.inference.sliding_window_prediction import (
    compute_gaussian,
    compute_steps_for_sliding_window,
)
from nnunetv2.utilities.file_path_utilities import (
    get_output_folder,
    check_workers_alive_and_busy,
)
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class nnUNetPredictor_extended(nnUNetPredictor):
    def __init__(
        self,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        perform_everything_on_gpu: bool = True,
        device: torch.device = torch.device("cuda"),
        verbose: bool = False,
        verbose_preprocessing: bool = False,
        allow_tqdm: bool = True,
        dataset_task_number: int = None,
        save_seg_masks: bool = False,
        csv_name: str = "results",
        mask_modality: str = None,
        modalities_to_keep: list = None,
        model_task_fullname: str = None,
        trainer_name: str = None,
        chk: str = None,
        config: str = None,
    ):
        super().__init__(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_gpu=perform_everything_on_gpu,
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose_preprocessing,
            allow_tqdm=allow_tqdm,
        )

        self.dataset_task_number = dataset_task_number
        self.save_seg_masks = save_seg_masks
        self.csv_name = csv_name
        self.mask_modality = mask_modality
        self.modalities_to_keep = modalities_to_keep
        self.trainer_name = trainer_name
        self.chk = chk
        self.config = config
        self.all_raw_dir = Path("/storage/nnUnet/nnUNet_raw/nnUNet_raw_data")
        self.raw_model_dir = self.all_raw_dir / model_task_fullname

        self.results_dir = (
            self.raw_model_dir
            / "results"
            / f'{self.dataset_task_number}_{self.trainer_name}_{self.chk.replace(".pth", "")}_{self.config}'
        )

    def prepare_filenames(self, raw_dataset_dir, data_split):
        if self.modalities_to_keep is not None:
            modality_dict = {
                j: i for i, j in self.data_dataset_json["channel_names"].items()
            }
            self.modality_idx_to_keep = [
                int(modality_dict[m]) for m in self.modalities_to_keep
            ]
        else:
            self.modality_idx_to_keep = None

        channel_names = list(self.data_dataset_json["channel_names"].keys())
        # TODO: extend fnc to also look into imagesTs folder
        gt_fpaths = []
        out_fpaths = []
        for fold, fold_split in enumerate(data_split):
            gt_out_fpaths = {}
            fold_out_fpaths = {}
            for phase, data_list in fold_split.items():
                new_data_list = []
                for fname in data_list:
                    new_data_list.append(
                        [
                            raw_dataset_dir
                            / "imagesTr"
                            / (fname + f"_{m.zfill(4)}.nii.gz")
                            for m in channel_names
                        ]
                    )

                fold_split[phase] = new_data_list
                gt_out_fpaths[phase] = [
                    raw_dataset_dir / "labelsTr" / (fname + f".nii.gz")
                    for fname in data_list
                ]
                fold_out_fpaths[phase] = [
                    self.results_dir / f"fold_{fold}" / phase / f"{fname}.nii.gz"
                    for fname in data_list
                ]
            gt_fpaths.append(gt_out_fpaths)
            out_fpaths.append(fold_out_fpaths)
        return data_split, gt_fpaths, out_fpaths

    def predict_one_by_one(self, list_of_lists, gt_list, ofilepaths, verbose=False):
        # check if we need to load gt seg
        concat_seg = "nnUNetTrainerSoftTissueOARs" in self.nnunet_trainer.__class__.__name__ or "nnUNetTrainerSmallOARs" in self.nnunet_trainer.__class__.__name__

        results = []

        preprocessor = self.configuration_manager.preprocessor_class(verbose=verbose)
        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        for idx in range(len(list_of_lists)):
            if concat_seg:
                seg_path = Path(
                    str(list_of_lists[idx][0])
                    .replace("/imagesT", "/labelsT")
                    .replace("_0000", "")
                )
                assert seg_path.exists()
            else:
                seg_path = None

            data, seg, data_properites = preprocessor.run_case(
                list_of_lists[idx],
                seg_path,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json,
            )

            if concat_seg:
                # move to torch tensor and add batch dimension
                data = torch.from_numpy(data)[None].contiguous().float()
                if 'justCT' in self.nnunet_trainer.__class__.__name__:
                    data = data[:, [0]]
                elif 'justMR' in self.nnunet_trainer.__class__.__name__:
                    data = data[:, [1]]
                seg = torch.from_numpy(seg)[None]

                # seg should be list (to be compatible with deep supervision formulation of target tensors)
                data, _ = self.nnunet_trainer.create_anchor_mask_and_prepare_target(
                    data,
                    [seg],
                )
                data = data[0].contiguous().float()
            else:
                data = torch.from_numpy(data).contiguous().float()

            prediction = self.predict_logits_from_preprocessed_data(data).cpu()

            segmentation_final = (
                convert_predicted_logits_to_segmentation_with_correct_shape(
                    prediction,
                    self.plans_manager,
                    self.configuration_manager,
                    label_manager,
                    data_properites,
                    return_probabilities=False,
                )
            )
            del prediction

            if concat_seg:
                segmentation_final_tmp = np.zeros_like(segmentation_final)
                for enum, val in enumerate(self.nnunet_trainer.new_target_organs_dict.values()):
                    segmentation_final_tmp[segmentation_final == (enum+1)] = val
                segmentation_final = segmentation_final_tmp

            sitk_pred = self.get_sitk_seg(
                segmentation_final, properties=data_properites
            )
            sitk_gt = sitk.ReadImage(str(gt_list[idx]))
            if self.save_seg_masks:
                ofilepaths[idx].parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(sitk_pred, str(ofilepaths[idx]))

            results.append(
                {
                    "gt": sitk_gt,
                    "pred": sitk_pred,
                    "gt_fpath": str(gt_list[idx]),
                    "pred_fpath": str(ofilepaths[idx]),
                }
            )

            # clear lru cache
            compute_gaussian.cache_clear()
            # clear device cache
            empty_cache(self.device)

        return results

    def get_pred(self, x):
        if self.modality_idx_to_keep is not None:
            x = x[:, self.modality_idx_to_keep]

        if self.config in ["3d_fullres_separatenorm"]:
            prediction = self.network(x, self.modality_idx_to_keep[0])
        elif self.config in ["3d_fullres_SeparateNormCMX", "3d_fullres_SepEncCMX"]:
            prediction = self.network(x[:, [0]], x[:, [1]])
        else:
            prediction = self.network(x)
        return prediction

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.get_pred(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert (
                max(mirror_axes) <= len(x.shape) - 3
            ), "mirror_axes does not match the dimension of the input!"

            num_predictons = 2 ** len(mirror_axes)
            if 0 in mirror_axes:
                prediction += torch.flip(self.get_pred(torch.flip(x, (2,))), (2,))
            if 1 in mirror_axes:
                prediction += torch.flip(self.get_pred(torch.flip(x, (3,))), (3,))
            if 2 in mirror_axes:
                prediction += torch.flip(self.get_pred(torch.flip(x, (4,))), (4,))
            if 0 in mirror_axes and 1 in mirror_axes:
                prediction += torch.flip(self.get_pred(torch.flip(x, (2, 3))), (2, 3))
            if 0 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.get_pred(torch.flip(x, (2, 4))), (2, 4))
            if 1 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.get_pred(torch.flip(x, (3, 4))), (3, 4))
            if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(
                    self.get_pred(torch.flip(x, (2, 3, 4))), (2, 3, 4)
                )
            prediction /= num_predictons
        return prediction

    def get_sitk_seg(self, seg: np.ndarray, properties: dict) -> None:
        assert (
            len(seg.shape) == 3
        ), "segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y"
        output_dimension = len(properties["sitk_stuff"]["spacing"])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]

        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8))
        itk_image.SetSpacing(properties["sitk_stuff"]["spacing"])
        itk_image.SetOrigin(properties["sitk_stuff"]["origin"])
        itk_image.SetDirection(properties["sitk_stuff"]["direction"])
        return itk_image

    def compute_metrics(self, data_dataset_json, settings_info, results):
        self.organs_labels_dict_pred = self.dataset_json["labels"]
        self.organs_labels_dict_gt = data_dataset_json["labels"]

        logging.debug(
            f"Found the following organs and labels GT dict: {self.organs_labels_dict_gt}"
        )
        logging.debug(
            f"Found the following organs and labels PRED dict: {self.organs_labels_dict_pred}"
        )
        self.compute_metrics_nnunet = compute_metrics_multilabel(
            organs_labels_dict_gt=self.organs_labels_dict_gt,
            organs_labels_dict_pred=self.organs_labels_dict_pred,
        )

        dfs = []
        for self.case in tqdm(results, total=len(results)):
            self.gt_fpath = self.case["gt_fpath"]
            self.pred_fpath = self.case["pred_fpath"]

            try:
                out_dict_tmp = self.compute_metrics_nnunet.execute(
                    fpath_gt=self.gt_fpath,
                    fpath_pred=self.pred_fpath,
                    img_gt=self.case["gt"],
                    img_pred=self.case["pred"],
                )
            except Exception as e:
                logging.error(f"Failed due to the following error: {e}")

            df = pd.DataFrame.from_dict(out_dict_tmp)
            for k, val in settings_info.items():
                df[k] = val
            dfs.append(df)

        try:
            sys.path.append(r"/media/medical/gasperp/projects")
            from utilities import utilities

            dfs = pd.concat(dfs, ignore_index=True)
            dfs = utilities.get_preserved_volume_ratio_info(
                dfs, dataset_dir=self.raw_model_dir
            )
            logging.info(
                f"Successfully appended `preserved_volume_ratio` to metrics dataframe."
            )
        except Exception as e:
            logging.error(
                f"Failed to append `preserved_volume_ratio` to metrics dataframe due to the following error: {e}"
            )
        finally:
            dfs = [dfs]

        try:
            csv_path = self.raw_model_dir / f"{self.csv_name}.csv"
            if os.path.exists(csv_path):
                logging.info(
                    f"Found existing .csv file on location {csv_path}, merging existing and new dataframe"
                )
                existing_df = [pd.read_csv(csv_path, index_col=0)] + dfs
                pd.concat(existing_df, ignore_index=True).to_csv(csv_path)
            else:
                pd.concat(dfs, ignore_index=True).to_csv(csv_path)
            logging.info(f"Successfully saved {self.csv_name}.csv file to {csv_path}")

        except Exception as e:
            logging.error(f"Failed due to the following error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Use this to run inference with nnU-Net. This function is used when "
        "you want to manually specify a folder containing a trained nnU-Net "
        "model. This is useful when the nnunet environment variables "
        "(nnUNet_results) are not set."
    )
    parser.add_argument(
        "-d",
        type=str,
        required=True,
        help="Dataset with which you would like to predict. You can specify either dataset name or id",
    )
    parser.add_argument(
        "-p",
        type=str,
        required=False,
        default="nnUNetPlans",
        help="Plans identifier. Specify the plans in which the desired configuration is located. "
        "Default: nnUNetPlans",
    )
    parser.add_argument(
        "-tr",
        type=str,
        required=False,
        default="nnUNetTrainer",
        help="What nnU-Net trainer class was used for training? Default: nnUNetTrainer",
    )
    parser.add_argument(
        "-c",
        type=str,
        default="3d_fullres",
        # choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
        help="nnU-Net configuration that should be used for prediction. Config must be located "
        "in the plans specified with -p",
    )
    parser.add_argument(
        "-f",
        nargs="+",
        type=str,
        required=False,
        default=(0, 1, 2, 3, 4),
        help="Specify the folds of the trained model that should be used for prediction. "
        "Default: (0, 1, 2, 3, 4)",
    )
    parser.add_argument(
        "-step_size",
        type=float,
        required=False,
        default=0.5,
        help="Step size for sliding window prediction. The larger it is the faster but less accurate "
        "the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.",
    )
    parser.add_argument(
        "--disable_tta",
        action="store_true",
        required=False,
        default=False,
        help="Set this flag to disable test time data augmentation in the form of mirroring. Faster, "
        "but less accurate inference. Not recommended.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set this if you like being talked to. You will have "
        "to be a good listener/reader.",
    )
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
        "multiple configurations.",
    )
    parser.add_argument(
        "--continue_prediction",
        action="store_true",
        help="Continue an aborted previous prediction (will not overwrite existing files)",
    )
    parser.add_argument(
        "-chk",
        type=str,
        required=False,
        default="checkpoint_best.pth",
        help="Name of the checkpoint you want to use. Default: checkpoint_final.pth",
    )
    parser.add_argument(
        "-npp",
        type=int,
        required=False,
        default=3,
        help="Number of processes used for preprocessing. More is not always better. Beware of "
        "out-of-RAM issues. Default: 3",
    )
    parser.add_argument(
        "-nps",
        type=int,
        required=False,
        default=3,
        help="Number of processes used for segmentation export. More is not always better. Beware of "
        "out-of-RAM issues. Default: 3",
    )
    parser.add_argument(
        "-prev_stage_predictions",
        type=str,
        required=False,
        default=None,
        help="Folder containing the predictions of the previous stage. Required for cascaded models.",
    )
    parser.add_argument(
        "-num_parts",
        type=int,
        required=False,
        default=1,
        help="Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one "
        "call predicts everything)",
    )
    parser.add_argument(
        "-part_id",
        type=int,
        required=False,
        default=0,
        help="If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with "
        "num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts "
        "5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible "
        "to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        required=False,
        help="Use this to set the device the inference should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
        "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!",
    )
    parser.add_argument(
        "--dataset_task_number",
        type=int,
        default=None,
        help="Only use this if dataset task number is different that model task number",
    )
    parser.add_argument(
        "--save_seg_masks",
        default=False,
        action="store_true",
        help="if this option is used, output segmentations are stored to out_dir",
    )
    parser.add_argument(
        "--phases_to_predict",
        nargs="+",
        type=str,
        choices=["test", "val", "train"],
        default=["val"],
        help="which phases to predict",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="results",
        help="",
    )
    parser.add_argument(
        "--mask_modality",
        type=str,
        default=None,
        choices=["CT", "MR_T1"],
        help="multiply one of the inputs with zero. Options: CT, MR_T1,... modality",
    )
    parser.add_argument(
        "--modalities_to_keep",
        nargs="+",
        type=str,
        default=None,
        choices=["CT", "MR_T1"],
        help="modality names to keep. Options: CT, MR_T1,...",
    )

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n"
    )

    args = parser.parse_args()
    args.f = [i if i == "all" else int(i) for i in args.f]

    args.dataset_task_number = (
        args.dataset_task_number if args.dataset_task_number else args.d
    )

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    model_task_fullname = maybe_convert_to_dataset_name(args.d)
    data_task_fullname = maybe_convert_to_dataset_name(
        args.dataset_task_number if args.dataset_task_number else args.d
    )
    model_raw_folder = Path(os.environ["nnUNet_raw"]) / model_task_fullname
    data_raw_folder = Path(os.environ["nnUNet_raw"]) / data_task_fullname
    data_preproc_folder = Path(os.environ["nnUNet_preprocessed"]) / data_task_fullname
    data_split = load_json(data_preproc_folder / "splits_final.json")

    # slightly passive agressive haha
    assert (
        args.part_id < args.num_parts
    ), "Do you even read the documentation? See nnUNetv2_predict -h."

    assert args.device in [
        "cpu",
        "cuda",
        "mps",
    ], f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}."
    if args.device == "cpu":
        # let's allow torch to use hella threads
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    predictor = nnUNetPredictor_extended(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_gpu=True,
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=False,
        dataset_task_number=args.dataset_task_number,
        save_seg_masks=args.save_seg_masks,
        csv_name=args.csv_name,
        mask_modality=args.mask_modality,
        modalities_to_keep=args.modalities_to_keep,
        model_task_fullname=model_task_fullname,
        trainer_name=args.tr,
        chk=args.chk,
        config=args.c,
    )

    from nnunetv2.run.run_training import get_trainer_from_args

    predictor.nnunet_trainer = get_trainer_from_args(
        args.d, args.c, args.f[0], args.tr, args.p, True, device=device
    )

    predictor.dataset_json = load_json(join(model_folder, "dataset.json"))
    data_dataset_json = load_json(join(data_raw_folder, "dataset.json"))
    predictor.data_dataset_json = data_dataset_json

    # TODO: extend to work for multiple folds
    (
        phases_list_of_lists,
        gt_list_of_lists,
        phases_ofilepaths,
    ) = predictor.prepare_filenames(data_raw_folder, data_split)

    if model_task_fullname != data_task_fullname:
        print(
            "Because model and dataset dir are different, doing prediction for all images in dataset dir."
        )
        args.phases_to_predict = ["val"]
        phases_list_of_lists = [
            dict(
                val=(phases_list_of_lists[0]["train"] + phases_list_of_lists[0]["val"])
            )
        ] * len(args.f)
        gt_list_of_lists = [
            dict(val=(gt_list_of_lists[0]["train"] + gt_list_of_lists[0]["val"]))
        ] * len(args.f)
        phases_ofilepaths = [
            dict(val=(phases_ofilepaths[0]["train"] + phases_ofilepaths[0]["val"]))
        ] * len(args.f)

    for fold in args.f:
        predictor.initialize_from_trained_model_folder(
            model_folder, [fold], checkpoint_name=args.chk
        )
        for phase in args.phases_to_predict:
            list_of_lists = phases_list_of_lists[fold].get(phase)
            gt_list = gt_list_of_lists[fold].get(phase)
            ofilepaths = phases_ofilepaths[fold].get(phase)
            if list_of_lists is None:
                continue
            settings_info = {
                "model_task_number": args.d,
                "model_task_name": model_task_fullname,
                "dataset_task_number": args.dataset_task_number,
                "dataset_task_name": data_task_fullname,
                "fold": fold,
                "phase": phase,
                "trainer_class": args.tr,
                "plans_name": args.p,
                "config": args.c,
                "checkpoint": args.chk,
                "TTA": not args.disable_tta,
                "masked_modality": args.mask_modality,
                "modalities_to_keep": None
                if args.modalities_to_keep is None
                else ", ".join(args.modalities_to_keep),
                "epoch": predictor.current_epoch,
            }
            results = predictor.predict_one_by_one(list_of_lists, gt_list, ofilepaths)

            predictor.compute_metrics(
                data_dataset_json, settings_info=settings_info, results=results
            )


if __name__ == "__main__":
    main()
