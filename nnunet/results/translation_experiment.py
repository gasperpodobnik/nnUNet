import logging
logging.info('start\n\n\n')
import argparse
import json
import os
import pickle
import shutil
import sys
from os.path import isdir, join
from pathlib import Path
from tqdm import tqdm
import subprocess
import nnunet.results.fcn as fcn
from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.model_restore import load_model_and_checkpoint_files
import numpy as np

import pandas as pd

sys.path.append(r"/media/medical/gasperp/projects")
sys.path.append(r"/media/medical/gasperp/projects/surface-distance")
from surface_distance import compute_metrics_deepmind, metrics

def main():
    # Set parser
    parser = argparse.ArgumentParser(
        prog="nnU-Net prediction generating script",
        description="Generate & evaluate predictions",
    )
    parser.add_argument(
        "-t",
        "--task_number",
        type=int,
        required=True,
        help="Task number of the model used for inference three digit number XXX that comes after TaskXXX_YYYYYY",
    )
    parser.add_argument(
        "--dataset_task_number",
        type=int,
        default=None,
        help="Only use this if dataset task number is different that model task number",
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=str,
        default=None,
        choices=["0", "1", "2", "3", "4", "all"],
        help="default is None (which means that script automatically determines fold if the is only one fold subfolder, otherwise raises error)",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=None,
        help="directory to store output csv file and predictions (if --save_seg_masks is enabled)",
    )
    parser.add_argument(
        "--save_seg_masks",
        default=False,
        action="store_true",
        help="if this option is used, output segmentations are stored to out_dir",
    )
    parser.add_argument(
        "-conf",
        "--configuration",
        type=str,
        default="3d_fullres",
        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
        help="nnU-Net configuration",
    )
    parser.add_argument(
        "-tr",
        "--trainer_class_name",
        type=str,
        default=None,
        help="nnU-Net trainer: default is None (which means that script automatically determines trainer class if the is only one trainer subfolder, otherwise raises error), common options are nnUNetTrainerV2, nnUNetTrainerV2_noMirroringAxis2",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="model_final_checkpoint",  # this means that 'model_final_checkpoint.model.pkl' is used for inference
        help="nnU-Net model to use: default is final model, but in case inference is done before model training is complete you may want to specifify 'model_best' or 'model_latest'",
    )
    parser.add_argument(
        "--phases_to_predict", 
        nargs="+",
        type=str,
        default=['test', 'val', 'train'],
        help="which phases to predict",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=str,
        default=None,
        help="if specified, GPU is utilized to speed up inference",
    )
    parser.add_argument(
        "--num_threads_preprocessing",
        type=int,
        default=1,
        help="nnUNet_predict parameter",
    )
    parser.add_argument(
        "--num_threads_nifti_save",
        type=int,
        default=1,
        help="nnUNet_predict parameter",
    )
    parser.add_argument(
        "--mode", type=str, default="normal", help="nnUNet_predict parameter",
    )
    parser.add_argument(
        "--csv_name", type=str, default="results", help="",
    )
    parser.add_argument(
        "--mask_modality", type=str, default=None, help="options: CT, MR,... modality",
    )
    parser.add_argument(
        "--disable_tta", 
        default=False,
        action="store_true",
        help="nnUNet_predict parameter",
    )
    parser.add_argument(
        "--inference_method", 
        default='one-by-one',
        type=str,
        help="method for getting predictions, can be: `one-by-one`, `folder`, `cmd`",
    )
    parser.add_argument(
        "--step_size", type=float, default=0.5, required=False, help="don't touch"
    )
    # step_size: When running sliding window prediction, the step size determines the distance between adjacent
    # predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
    # as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
    # predictions. step_size cannot be larger than 1!
    parser.add_argument(
        "--all_in_gpu",
        type=str,
        default="None",
        required=False,
        help="can be None, False or True",
    )

    # running in terminal
    args = vars(parser.parse_args())

    assert len(args['phases_to_predict']) > 0, 'there should be at least one phase'
    
    all_in_gpu = args["all_in_gpu"]
    assert all_in_gpu in ["None", "False", "True"]
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False
        
    inference_on_unseen_dataset = True
    if args["dataset_task_number"] is None:
        args["dataset_task_number"] = args["task_number"]
        inference_on_unseen_dataset = False

    # paths definition
    nnUNet_raw_data_base_dir = os.environ["nnUNet_raw_data_base"]
    nnUNet_preprocessed_dir = os.environ["nnUNet_preprocessed"]
    nnUNet_trained_models_dir = os.environ["RESULTS_FOLDER"]
    nnUNet_configuration_dir = join(
        nnUNet_trained_models_dir, "nnUNet", args["configuration"]
    )
    base_nnunet_dir_on_medical = "/media/medical/projects/head_and_neck/nnUnet"
    csv_name = args["csv_name"]

    ## checkers for input parameters
    # input task
    existing_tasks_for_models = {
        int(i.split("_")[0][-3:]): join(nnUNet_configuration_dir, i)
        for i in os.listdir(nnUNet_configuration_dir)
        if i.startswith("Task") and os.path.isdir(join(nnUNet_configuration_dir, i))
    }
    existing_tasks_for_datasets = {
        int(i.split("_")[0][-3:]): join(base_nnunet_dir_on_medical, i)
        for i in os.listdir(base_nnunet_dir_on_medical)
        if i.startswith("Task") and os.path.isdir(join(base_nnunet_dir_on_medical, i))
    }
    existing_tasks_for_datasets = {**existing_tasks_for_models, **existing_tasks_for_datasets}
    assert (
        args["task_number"] in existing_tasks_for_models.keys()
    ), f"Could not find task num.: {args['task_number']}. Found the following task/directories: {existing_tasks_for_models}"
    assert (
        args["dataset_task_number"] in existing_tasks_for_datasets.keys()
    ), f"Could not find task num.: {args['dataset_task_number']}. Found the following task/directories: {existing_tasks_for_datasets}"

    model_task_dir = existing_tasks_for_models[args["task_number"]]
    dataset_task_dir = existing_tasks_for_datasets[args["dataset_task_number"]]
    # e.g.: '/storage/nnUnet/nnUNet_trained_models/nnUNet/3d_fullres/Task152_onkoi-2019-batch-1-and-2-both-modalities-biggest-20-organs-new'
    model_task_name = Path(model_task_dir).name
    dataset_task_name = Path(dataset_task_dir).name
    # e.g.: task_name = 'Task152_onkoi-2019-batch-1-and-2-both-modalities-biggest-20-organs-new'

    ## checkers for input parameters
    # nnunet trainer class
    trainer_classes_list = [i.split("__")[0] for i in os.listdir(model_task_dir)]
    assert len(trainer_classes_list) > 0, f"no trainer subfolders found in {model_task_dir}"
    if args["trainer_class_name"] is None:
        if len(trainer_classes_list) > 1:
            ValueError(
                f"Cannot automatically determine trainer class name, since multiple trainer class folders were found in {model_task_dir}. \nPlease specfiy exact '--trainer_class_name'"
            )
        else:
            args["trainer_class_name"] = trainer_classes_list[0]

    ## checkers for input parameters
    # nnunet plans list
    # determine which plans version was used, raise error if multiple plans exist
    plans_list = [
        i.split("__")[-1]
        for i in os.listdir(model_task_dir)
        if i.startswith(f"{args['trainer_class_name']}__")
    ]
    assert (
        len(plans_list) == 1
    ), f"multiple trainer_classes_and_plans dirs found {plans_list}, please specify which to use"
    args["plans_name"] = plans_list[0]
    args[
        "trainer_classes_and_plans_dir_name"
    ] = f"{args['trainer_class_name']}__{args['plans_name']}"

    trainer_classes_and_plans_dir = join(
        model_task_dir, args["trainer_classes_and_plans_dir_name"]
    )

    ## checkers for input parameters
    # fold
    available_folds = [
        i
        for i in os.listdir(trainer_classes_and_plans_dir)
        if os.path.isdir(join(trainer_classes_and_plans_dir, i))
    ]
    if "gt_niftis" in available_folds:
        available_folds.remove("gt_niftis")
    assert (
        len(available_folds) > 0
    ), f"no fold subfolders found in {trainer_classes_and_plans_dir}"

    if args["fold"] is None:
        if len(available_folds) > 1:
            ValueError(
                f"Cannot automatically determine fold, since multiple folds were found in {trainer_classes_and_plans_dir}. \nPlease specfiy exact '--fold'"
            )
        else:
            get_fold_num = lambda s: int(s.split("_")[-1]) if s != "all" else s
            args["fold"] = get_fold_num(available_folds[0])
    if args["fold"] != "all":
        # if args['fold'] is 0/1/2/3/4, convert it to 'fold_X' else keep 'all'
        args['fold'] = int(args['fold'])
        args["fold_str"] = f"fold_{args['fold']}"
    else:
        args["fold_str"] = args["fold"]
    assert (
        args["fold_str"] in available_folds
    ), f"--fold {args['fold']} is not a valid options, available_folds are: {available_folds}"

    ## checkers for input parameters
    # nnunet model checkpoint to be used for inference

    models_checkpoints_dir = join(trainer_classes_and_plans_dir, args["fold_str"])
    models_checkpoints_dir_files = os.listdir(models_checkpoints_dir)
    assert any(
        [args["checkpoint_name"] in i for i in models_checkpoints_dir_files]
    ), f"--checkpoint_name {args['checkpoint_name']} is not a valid options, checkpoint_name should be a file in {models_checkpoints_dir}. Files in this directory are: {models_checkpoints_dir_files}"

    ## data paths retrieval
    # get dict, dict of dict of filepaths: {'train': {img_name: {'images': {modality0: fpath, ...}, 'label': fpath} ...}, ...}
    # load dataset json
    with open(join(nnUNet_preprocessed_dir, model_task_name, "dataset.json"), "r") as fp:
        model_dataset_json_dict = json.load(fp)
    # create modalities dict
    four_digit_ids = {
        m: str.zfill(str(int(i)), 4) for i, m in model_dataset_json_dict["modality"].items()
    }
    MODALITY_to_mask=int(four_digit_ids[args.get('mask_modality')]) if args.get('mask_modality') else args.get('mask_modality')

    # get image paths with modality four digit id
    raw_data_dir = join(nnUNet_raw_data_base_dir, "nnUNet_raw_data", dataset_task_name)
    if not os.path.exists(raw_data_dir):
        raw_data_dir = join(base_nnunet_dir_on_medical, dataset_task_name)
    assert os.path.exists(raw_data_dir), ValueError(f'raw dataset dir does now exist {raw_data_dir}')
    
    with open(join(raw_data_dir, "dataset.json"), "r") as fp:
        dataset_dataset_json_dict = json.load(fp)
    
    img_modality_fpaths_dict = lambda fname, dir_name: {
        modality: join(raw_data_dir, f"images{dir_name}", fname + f"_{m_id}.nii.gz")
        for modality, m_id in four_digit_ids.items()
    }

    # generate label path
    labels_fpath = lambda fname, dir_name: join(
        raw_data_dir, f"labels{dir_name}", fname + ".nii.gz"
    )
    # generate images dict {modality: img_path}
    splits_iterator = lambda fname_list, dir_name="Tr": {
        img_name: {
            "images": img_modality_fpaths_dict(img_name, dir_name=dir_name),
            "label": labels_fpath(img_name, dir_name=dir_name),
        }
        for img_name in fname_list
    }

    # create dict with paths split on train, val (if fold not 'all') and test
    splits_final_dict = {}
    if 'test' in args['phases_to_predict']:
        splits_final_dict["test"] = splits_iterator(
            [Path(i).name[: -len(".nii.gz")] for i in model_dataset_json_dict["test"]],
            dir_name="Ts",
        )
    if 'train' in args['phases_to_predict'] or 'val' in args['phases_to_predict']:
        if args["fold"] != "all" and (not inference_on_unseen_dataset):
            with open(
                join(nnUNet_preprocessed_dir, model_task_name, "splits_final.pkl"), "rb"
            ) as f:
                _dict = pickle.load(f)
            splits_final_dict["train"] = splits_iterator(_dict[int(args["fold"])]["train"])
            splits_final_dict["val"] = splits_iterator(_dict[int(args["fold"])]["val"])
        else:
            splits_final_dict["train"] = splits_iterator(
                [
                    Path(_dict["image"]).name[: -len(".nii.gz")]
                    for _dict in model_dataset_json_dict["training"]
                ]
            )

    images_source_dirs = []

    if 'test' in args['phases_to_predict']:
        images_source_dirs.append({'phase': 'test', 'img_dir': join(raw_data_dir, "imagesTs"), 'gt_dir': join(raw_data_dir, "labelsTs")})

    if 'train' in args['phases_to_predict'] or 'val' in args['phases_to_predict']:
        images_source_dirs.append({'phase': 'train', 'img_dir': join(raw_data_dir, "imagesTr"), 'gt_dir': join(raw_data_dir, "labelsTr")})

    # config_str = f"FOLD-{args['fold']}-{args['trainer_class_name']}-{args['plans_name']}_CHK-{args['checkpoint_name']}-{args['dataset_task_number']}_TTA-{not args['disable_tta']}_STEP-{args['step_size']}"
    config_str = f"{args['dataset_task_number']}"
    if args.get('mask_modality'):
        config_str+=f"_masked-{args.get('mask_modality')}"
    logging.info(f"settings info: {config_str}")
    if args["out_dir"] is None:
        args["out_dir"] = join(base_nnunet_dir_on_medical, model_task_name, "patch_registration_experiment")
    elif not args["out_dir"].startswith('/'):
        args["out_dir"] = join(base_nnunet_dir_on_medical, model_task_name, "patch_registration_experiment", args["out_dir"])
    os.makedirs(args["out_dir"], exist_ok=True)

    # prepare directories in case predicted segmentations are to be saved
    if args["save_seg_masks"]:
        pred_seg_out_dir = join(args["out_dir"], config_str)
        out_dirs = {}

        if 'test' in args['phases_to_predict']:
            out_dirs["test"] = join(pred_seg_out_dir, "test")
            os.makedirs(out_dirs["test"], exist_ok=True)
            
        if 'train' in args['phases_to_predict']:
            out_dirs["train"] = join(pred_seg_out_dir, "train")
            os.makedirs(out_dirs["train"], exist_ok=True)
            
        if 'val' in args['phases_to_predict']:
            assert "val" in splits_final_dict.keys(), 'there are no images in the validation set (if fold options is `all`, set --phases_to_predict to just `test` `train`)'
            out_dirs["val"] = join(pred_seg_out_dir, "val")
            os.makedirs(out_dirs["val"], exist_ok=True)

    inverse_splits_final_dict  = {cid: phase for phase, cases_dict in splits_final_dict.items() for cid, _ in cases_dict.items()}
    
    model_folder_name = join(
                nnUNet_configuration_dir,
                trainer_classes_and_plans_dir
            )
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), (
        "model output folder not found. Expected: %s" % model_folder_name
    )
    
    N=5
    distances = [0] + (2**np.array([1, 2, 3, 4, 5, 6])).tolist()
    
    experiment = 'registration'
    reged_organ = 'Glnd_Submand_R'
    rigid=True
    deformable=True
    
    successfully_predicted = []
    try:
        if args['inference_method'] == 'one-by-one':
            assert args["mode"] == 'normal', NotImplementedError('current implementation for one-by-one_method supports only normal mode')
            from nnunet.inference.predict import predict_cases, check_input_folder_and_return_caseIDs, subfiles, load_pickle, preprocess_multithreaded, save_segmentation_nifti_from_softmax
            import torch

            expected_num_modalities = load_pickle(join(model_folder_name, "plans.pkl"))["num_modalities"]
            for _dict_tmp in images_source_dirs:
                phase = _dict_tmp['phase']
                img_dir = _dict_tmp['img_dir']
                case_ids = check_input_folder_and_return_caseIDs(
                    img_dir, expected_num_modalities
                )
                case_ids = [cid for cid in case_ids if inverse_splits_final_dict[cid] == phase]
                
                print(f'\n\nFound {len(case_ids)} cases in phase {phase}\n\n')
                
                output_files = [join(out_dirs[phase], cid + ".nii.gz") for cid in case_ids]
                gt_files = [join(_dict_tmp['gt_dir'], cid + ".nii.gz") for cid in case_ids]
                all_files = subfiles(img_dir, suffix=".nii.gz", join=False, sort=True)
                list_of_lists = [
                    [
                        join(img_dir, i)
                        for i in all_files
                        if i[: len(j)].startswith(j) and len(i) == (len(j) + 12)
                    ]
                    for j in case_ids
                ]

                for input_filename, output_filename, gt_filename in zip(list_of_lists, output_files, gt_files):
                    step_size = args["step_size"]
                    do_tta=not args["disable_tta"]
                    try:
                        torch.cuda.empty_cache()
                        if not os.path.exists(output_filename):
                            # predict_cases(
                            #     model=model_folder_name,
                            #     list_of_lists=[input_filename],
                            #     output_filenames=[output_filename],
                            #     folds=[args["fold"]],
                            #     save_npz=False,
                            #     num_threads_preprocessing=args["num_threads_preprocessing"],
                            #     num_threads_nifti_save=args["num_threads_nifti_save"],
                            #     segs_from_prev_stage=None,
                            #     do_tta=do_tta,
                            #     mixed_precision=not False,
                            #     overwrite_existing=False,
                            #     all_in_gpu=all_in_gpu,
                            #     step_size=step_size,
                            #     checkpoint_name=args["checkpoint_name"],
                            #     segmentation_export_kwargs=None,
                            #     MODALITY_to_mask=MODALITY_to_mask
                            # )
                            
                            # ------------------------------------------------------------------------------
                            model=model_folder_name
                            list_of_lists=[input_filename]
                            output_filenames=[output_filename]
                            folds=[args["fold"]]
                            save_npz=False
                            num_threads_preprocessing=args["num_threads_preprocessing"]
                            num_threads_nifti_save=args["num_threads_nifti_save"]
                            segs_from_prev_stage=None
                            do_tta=do_tta
                            mixed_precision=not False
                            overwrite_existing=False
                            all_in_gpu=all_in_gpu
                            step_size=step_size
                            checkpoint_name=args["checkpoint_name"]
                            segmentation_export_kwargs=None
                            MODALITY_to_mask=MODALITY_to_mask
                            
                            assert len(list_of_lists) == len(output_filenames)
                            if segs_from_prev_stage is not None:
                                assert len(segs_from_prev_stage) == len(output_filenames)

                            pool = Pool(num_threads_nifti_save)
                            results = []

                            cleaned_output_files = []
                            for o in output_filenames:
                                dr, f = os.path.split(o)
                                if len(dr) > 0:
                                    maybe_mkdir_p(dr)
                                if not f.endswith(".nii.gz"):
                                    f, _ = os.path.splitext(f)
                                    f = f + ".nii.gz"
                                cleaned_output_files.append(join(dr, f))

                            if not overwrite_existing:
                                print("number of cases:", len(list_of_lists))
                                # if save_npz=True then we should also check for missing npz files
                                not_done_idx = [
                                    i
                                    for i, j in enumerate(cleaned_output_files)
                                    if (not isfile(j)) or (save_npz and not isfile(j[:-7] + ".npz"))
                                ]

                                cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
                                list_of_lists = [list_of_lists[i] for i in not_done_idx]
                                if segs_from_prev_stage is not None:
                                    segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

                                print(
                                    "number of cases that still need to be predicted:",
                                    len(cleaned_output_files),
                                )

                            print("emptying cuda cache")
                            torch.cuda.empty_cache()

                            print("loading parameters for folds,", folds)
                            trainer, params = load_model_and_checkpoint_files(
                                model, folds, mixed_precision=mixed_precision, checkpoint_name=checkpoint_name
                            )
                            
                            if len(params) > 1:
                                raise NotImplementedError('num params greater than 1, this is not implemented yet, see original nnunet code')

                            if segmentation_export_kwargs is None:
                                if "segmentation_export_params" in trainer.plans.keys():
                                    force_separate_z = trainer.plans["segmentation_export_params"][
                                        "force_separate_z"
                                    ]
                                    interpolation_order = trainer.plans["segmentation_export_params"][
                                        "interpolation_order"
                                    ]
                                    interpolation_order_z = trainer.plans["segmentation_export_params"][
                                        "interpolation_order_z"
                                    ]
                                else:
                                    force_separate_z = None
                                    interpolation_order = 1
                                    interpolation_order_z = 0
                            else:
                                force_separate_z = segmentation_export_kwargs["force_separate_z"]
                                interpolation_order = segmentation_export_kwargs["interpolation_order"]
                                interpolation_order_z = segmentation_export_kwargs["interpolation_order_z"]

                            print("starting preprocessing generator")
                            preprocessing = preprocess_multithreaded(
                                trainer,
                                list_of_lists,
                                cleaned_output_files,
                                num_threads_preprocessing,
                                segs_from_prev_stage,
                            )
                            print("starting prediction...")
                            all_output_files = []
                            for preprocessed in preprocessing:
                                output_filename, (d, dct) = preprocessed
                                all_output_files.append(all_output_files)
                                if isinstance(d, str):
                                    data = np.load(d)
                                    os.remove(d)
                                    d = data
                                    
                                    
                                
                                
                                root_results_dir = join(args["out_dir"], f'{experiment}_rigid-{rigid}_deformable-{deformable}_{reged_organ}')
                                os.makedirs(root_results_dir, exist_ok=True)
                                final_patch_size = np.array([192, 192, 40])
                                rough_patch_size = final_patch_size + np.array([20, 20, 6])
                                organs_labels_dict = {j: int(i) for i, j in dataset_dataset_json_dict['labels'].items()}
                                organ_lbl_int = organs_labels_dict[reged_organ]
                                
                                
                                filename = Path(gt_filename).name.split('_')[-1].split('.')[0]
                                import SimpleITK as sitk
                                seg_sitk = sitk.ReadImage(gt_filename)
                                
                                default_pixel_value_ct = float(d[0].min())
                                default_pixel_value_mr = float(d[1].min())
                                ct_sitk = sitk.GetImageFromArray(d[0])
                                ct_sitk.CopyInformation(seg_sitk)
                                mr_sitk = sitk.GetImageFromArray(d[1])
                                mr_sitk.CopyInformation(seg_sitk)
                                
                                # compute organ bbox
                                label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
                                label_shape_filter.Execute(seg_sitk)
                                bbox_idx = np.array(label_shape_filter.GetBoundingBox(organ_lbl_int))
                                bbox_start_idx, bbox_size = bbox_idx[:3], bbox_idx[3:]
                                bbox_end_idx = bbox_start_idx + bbox_size - 1
                                bbox_center_idx = (bbox_start_idx + bbox_end_idx)/2
                                

                                
                                
                                
                                
                                
                                
                                # --------------------------------------------------------------------------------
                                # extract small patch around bbox of the organ of interest and register it rigidly
                                bbox_extension = (bbox_size/4).astype(int)
                                reg_patch_size = bbox_size + 2*bbox_extension
                                reg_patch_start_idx = bbox_start_idx - bbox_extension
                                
                                if np.any(reg_patch_size < 4):
                                    continue
                                
                                reg_resample_filter = sitk.ResampleImageFilter()
                                reg_resample_filter.SetReferenceImage(seg_sitk)
                                reg_resample_filter.SetOutputOrigin(seg_sitk.TransformIndexToPhysicalPoint(reg_patch_start_idx.tolist()))
                                reg_resample_filter.SetSize(reg_patch_size.tolist())
                                reg_resample_filter.SetInterpolator(sitk.sitkBSpline)
                                reg_resample_filter.SetDefaultPixelValue(default_pixel_value_ct)
                                ct_rigid_reg_patch_sitk = reg_resample_filter.Execute(ct_sitk)
                                reg_resample_filter.SetDefaultPixelValue(default_pixel_value_mr)
                                mr_rigid_reg_patch_sitk = reg_resample_filter.Execute(mr_sitk)
                                
                                rigid_transform = fcn.register_patches(fixed_patch=ct_rigid_reg_patch_sitk, moving_patch=mr_rigid_reg_patch_sitk, rigid=rigid, deformable=False, root_dir=root_results_dir)
                                
                                rigid_reg_mr_sitk = fcn.apply_transform(mr_sitk, rigid_transform)
                                reg_mr_sitk = rigid_reg_mr_sitk
                                # --------------------------------------------------------------------------------
                                
                                if deformable:
                                    rough_patch_start_idx = (bbox_center_idx - rough_patch_size/2.0).astype(int)
                                    rough_resample_filter = sitk.ResampleImageFilter()
                                    rough_resample_filter.SetReferenceImage(seg_sitk)
                                    rough_resample_filter.SetOutputOrigin(seg_sitk.TransformIndexToPhysicalPoint(rough_patch_start_idx.tolist()))
                                    rough_resample_filter.SetSize(rough_patch_size.tolist())
                                    rough_resample_filter.SetInterpolator(sitk.sitkBSpline)
                                    rough_resample_filter.SetDefaultPixelValue(default_pixel_value_ct)
                                    ct_deformable_reg_patch_sitk = rough_resample_filter.Execute(ct_sitk)
                                    rough_resample_filter.SetDefaultPixelValue(default_pixel_value_mr)
                                    mr_deformable_reg_patch_sitk = rough_resample_filter.Execute(rigid_reg_mr_sitk)
                                    
                                    deformable_transform = fcn.register_patches(fixed_patch=ct_deformable_reg_patch_sitk, moving_patch=mr_deformable_reg_patch_sitk, rigid=False, deformable=True, root_dir=root_results_dir)
                                    
                                    deformable_reg_mr_sitk_patch = fcn.apply_transform(mr_deformable_reg_patch_sitk, deformable_transform)
                                    reg_mr_sitk = deformable_reg_mr_sitk_patch
                                
                                final_patch_start_idx = (bbox_center_idx - final_patch_size/2.0).astype(int)
                                final_resample_filter = sitk.ResampleImageFilter()
                                final_resample_filter.SetReferenceImage(seg_sitk)
                                final_resample_filter.SetOutputOrigin(seg_sitk.TransformIndexToPhysicalPoint(final_patch_start_idx.tolist()))
                                final_resample_filter.SetSize(final_patch_size.tolist())
                                final_resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
                                final_resample_filter.SetDefaultPixelValue(0)
                                seg_final_patch_sitk = final_resample_filter.Execute(seg_sitk)
                                final_resample_filter.SetInterpolator(sitk.sitkBSpline)
                                
                                final_resample_filter.SetDefaultPixelValue(default_pixel_value_ct)
                                ct_final_patch_sitk = final_resample_filter.Execute(ct_sitk)
                                
                                final_resample_filter.SetDefaultPixelValue(default_pixel_value_mr)
                                rigid_reg_mr_final_patch_sitk = final_resample_filter.Execute(rigid_reg_mr_sitk)
                                reg_mr_final_patch_sitk = final_resample_filter.Execute(reg_mr_sitk)
                                unreg_mr_final_patch_sitk = final_resample_filter.Execute(mr_sitk)



                                filepath_root = join(root_results_dir, filename + '_')
                                for ee, t in enumerate(rigid_transform):
                                    sitk.WriteParameterFile(t, filepath_root + str(ee) + '_tp_map_rigid.txt')
                                if deformable:
                                    for ee, t in enumerate(deformable_transform):
                                        sitk.WriteParameterFile(t, filepath_root + str(ee) + '_tp_map_deformable.txt')
                                sitk.WriteImage(ct_final_patch_sitk, filepath_root + 'ct_final_patch_sitk.nii.gz')
                                sitk.WriteImage(unreg_mr_final_patch_sitk, filepath_root + 'unreg_mr_final_patch_sitk.nii.gz')
                                sitk.WriteImage(reg_mr_final_patch_sitk, filepath_root + 'reg_mr_final_patch_sitk.nii.gz')
                                sitk.WriteImage(rigid_reg_mr_final_patch_sitk, filepath_root + 'rigid_reg_mr_final_patch_sitk.nii.gz')
                                sitk.WriteImage(sitk.Cast(seg_final_patch_sitk==organ_lbl_int, sitk.sitkUInt8), filepath_root + 'seg_final_patch_sitk.nii.gz')

                                    
                                # translate_filter = sitk.ResampleImageFilter()
                                # translate_filter.SetReferenceImage(seg_sitk)
                                # translate_filter.SetInterpolator(sitk.sitkBSpline)
                                # translate_filter.SetDefaultPixelValue(default_pixel_value_mr)
                                
                                # ct_patch = sitk.GetArrayFromImage(ct_final_patch_sitk)
                                dct['crop_bbox'] = None #[[i, j] for i, j in zip(real_start[::-1], real_end[::-1])]
                                dct['size_after_cropping'] = final_patch_size[::-1]
                                dct['itk_origin'] = ct_final_patch_sitk.GetOrigin()
                                
                                to_predict = [[unreg_mr_final_patch_sitk, False, False], [rigid_reg_mr_final_patch_sitk, rigid, False]]
                                if deformable:
                                    to_predict.append([reg_mr_final_patch_sitk, rigid, deformable])
                                
                                
                                # output_filename_orig = output_filename
                                # for distance in distances:
                                # fcn.transform_and_get_patch(distance, N, mr_sitk, rough_resample_filter, roi_filter)
                                for enum, (mr_patch, isRigid, isDeformable) in enumerate(to_predict):
                                    # print(distance, translation_params)
                                    # output_filename = output_filename_orig.replace('.nii.gz', f'_D{distance}_{enum}.nii.gz')
                                    output_filename = filepath_root + f'_RED-SEG_rigid-{isRigid}_deformable-{isDeformable}.nii.gz'
                                    d = np.stack((sitk.GetArrayFromImage(ct_final_patch_sitk), sitk.GetArrayFromImage(mr_patch)))
                                    
                                    if MODALITY_to_mask is not None:
                                        d[MODALITY_to_mask] = 0
                                        print(f'MASKING modality with index {MODALITY_to_mask}')

                                    print("predicting", output_filename)
                                    trainer.load_checkpoint_ram(params[0], False)
                                    softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
                                        d,
                                        do_mirroring=False,
                                        mirror_axes=[],
                                        use_sliding_window=False,
                                        all_in_gpu=all_in_gpu,
                                        mixed_precision=mixed_precision,
                                    )[1]


                                    transpose_forward = trainer.plans.get("transpose_forward")
                                    if transpose_forward is not None:
                                        transpose_backward = trainer.plans.get("transpose_backward")
                                        softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

                                    if save_npz:
                                        npz_file = output_filename[:-7] + ".npz"
                                    else:
                                        npz_file = None

                                    if hasattr(trainer, "regions_class_order"):
                                        region_class_order = trainer.regions_class_order
                                    else:
                                        region_class_order = None

                                    """There is a problem with python process communication that prevents us from communicating objects 
                                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
                                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
                                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
                                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
                                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
                                    filename or np.ndarray and will handle this automatically"""
                                    bytes_per_voxel = 4
                                    if all_in_gpu:
                                        bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
                                    if np.prod(softmax.shape) > (
                                        2e9 / bytes_per_voxel * 0.85
                                    ):  # * 0.85 just to be save
                                        print(
                                            "This output is too large for python process-process communication. Saving output temporarily to disk"
                                        )
                                        np.save(output_filename[:-7] + ".npy", softmax)
                                        softmax = output_filename[:-7] + ".npy"

                                    results.append(
                                        pool.starmap_async(
                                            save_segmentation_nifti_from_softmax,
                                            (
                                                (
                                                    softmax,
                                                    output_filename,
                                                    dct,
                                                    interpolation_order,
                                                    region_class_order,
                                                    None,
                                                    None,
                                                    npz_file,
                                                    None,
                                                    force_separate_z,
                                                    interpolation_order_z,
                                                ),
                                            ),
                                        )
                                    )
                                    successfully_predicted.append({'fname': Path(input_filename[0]).name.replace('.nii.gz', ''), 
                                                    'pred_fpath': output_filename, 
                                                    'gt_fpath': gt_filename, 
                                                    'phase': phase, 
                                                    'rigid': isRigid, 
                                                    'deformable': isDeformable,
                                                    'reged_organ': reged_organ
                                                    })
                            print("inference done. Now waiting for the segmentation export to finish...")
                            _ = [i.get() for i in results]
                            # now apply postprocessing
                            # first load the postprocessing properties if they are present. Else raise a well visible warning
                            pool.close()
                            pool.join()
                            # ------------------------------------------------------------------------------
                        else:
                            logging.info('Skipping inference, because output file exists')
                        
                    except Exception as e:
                        logging.error(f"Failed due to predict case {input_filename} due to the following error: {e}")
        else:
            raise NotImplementedError('Unknown --inference_method was specified')

    except Exception as e:
        logging.error(f"Failed due to the following error: {e}")
        # shutil.rmtree(output_seg_dir)
        sys.exit()

    try:
        organs_labels_dict_pred = {
            organ: int(lbl) for lbl, organ in model_dataset_json_dict["labels"].items()
        }
        organs_labels_dict_gt = {
            organ: int(lbl) for lbl, organ in dataset_dataset_json_dict["labels"].items()
        }
        logging.debug(f"Found the following organs and labels GT dict: {organs_labels_dict_gt}")
        logging.debug(f"Found the following organs and labels PRED dict: {organs_labels_dict_pred}")
        compute_metrics = compute_metrics_deepmind(
            organs_labels_dict_gt=organs_labels_dict_gt, organs_labels_dict_pred=organs_labels_dict_pred
        )
        settings_info = {
            "model_task_number": args["task_number"],
            "model_task_name": model_task_name,
            "dataset_task_number": args["dataset_task_number"],
            "dataset_task_name": dataset_task_name,
            "fold": args["fold"],
            "trainer_class": args["trainer_class_name"],
            "plans_name": args["plans_name"],
            "checkpoint": args["checkpoint_name"],
            "prediction_mode": args["mode"],
            "TTA": not args['disable_tta'],
            "masked_modality": args.get('mask_modality')
        }
        dfs = []

        for case in tqdm(
            successfully_predicted, total=len(successfully_predicted)
        ):
            settings_info["phase"] = case['phase']
            settings_info['distance']=case.get('distance')
            settings_info['rigid']=case.get('rigid')
            settings_info['deformable']=case.get('deformable')
            settings_info['translation_params']=str(case.get('translation_params'))
            settings_info['patch_enum']=case.get('enum')
            settings_info['reged_organ']=case.get('reged_organ')
            
            # settings_info["fname"] = case['fname']

            gt_fpath = case['gt_fpath']
            pred_fpath = case['pred_fpath']
            
            img_pred = sitk.ReadImage(pred_fpath)
            img_gt = sitk.ReadImage(gt_fpath)
            
            # crop gt image to same patch size as pred image
            final_resample_filter = sitk.ResampleImageFilter()
            final_resample_filter.SetReferenceImage(img_pred)
            final_resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
            final_resample_filter.SetDefaultPixelValue(0)
            img_gt = final_resample_filter.Execute(img_gt)

            out_dict_tmp = compute_metrics.execute(
                fpath_gt=gt_fpath, fpath_pred=pred_fpath, img_gt=img_gt, img_pred=img_pred
            )
            df = pd.DataFrame.from_dict(out_dict_tmp)
            for k, val in settings_info.items():
                df[k] = val
            dfs.append(df)

        csv_path = join(args["out_dir"], f"{csv_name}.csv")
        if os.path.exists(csv_path):
            logging.info(
                f"Found existing .csv file on location {csv_path}, merging existing and new dataframe"
            )
            existing_df = [pd.read_csv(csv_path, index_col=0)] + dfs
            pd.concat(existing_df, ignore_index=True).to_csv(csv_path)
        else:
            pd.concat(dfs, ignore_index=True).to_csv(csv_path)
        logging.info(f"Successfully saved {csv_name}.csv file to {csv_path}")
        
    except Exception as e:
        logging.error(f"Failed due to the following error: {e}")

    finally:
        sys.exit()


if __name__ == "__main__":
    main()
