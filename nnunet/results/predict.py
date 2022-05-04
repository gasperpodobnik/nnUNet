import numpy as np
import argparse
import copy
import glob
import json
import logging
import os
import pickle
import random
import shutil
import sys
import tempfile
from os.path import isdir, join
from pathlib import Path
from tqdm.std import tqdm
import subprocess

import pandas as pd

# sys.path.append(r"/media/medical/gasperp/projects")
# import utilities
# sys.path.append(r"/media/medical/gasperp/projects/surface-distance")
from surface_distance import compute_metrics_deepmind


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
        help="three digit number XXX that comes after TaskXXX_YYYYYY",
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
        help="nnU-Net model to use: default is final model, but in case inference is done before model training is complete you may want to specifify 'model_best.model.pkl' or sth else",
    )
    parser.add_argument(
        "--just_test", 
        default=False,
        action="store_true",
        help="just test",
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
        default=4,
        help="nnUNet_predict parameter",
    )
    parser.add_argument(
        "--mode", type=str, default="normal", help="nnUNet_predict parameter",
    )
    parser.add_argument(
        "--disable_tta", 
        default=False,
        action="store_true",
        help="nnUNet_predict parameter",
    )
    parser.add_argument(
        "--direct_method", 
        default=False,
        action="store_true",
        help="method for getting predictions",
    )
    parser.add_argument(
        "--step_size", type=float, default=0.5, required=False, help="don't touch"
    )
    parser.add_argument(
        "--all_in_gpu",
        type=str,
        default="None",
        required=False,
        help="can be None, False or True",
    )

    # running in terminal
    args = vars(parser.parse_args())
    
    all_in_gpu = args["all_in_gpu"]
    assert all_in_gpu in ["None", "False", "True"]
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    all_in_gpu = args["all_in_gpu"]
    assert all_in_gpu in ["None", "False", "True"]
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    # paths definition
    nnUNet_raw_data_base_dir = os.environ["nnUNet_raw_data_base"]
    nnUNet_preprocessed_dir = os.environ["nnUNet_preprocessed"]
    nnUNet_trained_models_dir = os.environ["RESULTS_FOLDER"]
    nnUNet_configuration_dir = join(
        nnUNet_trained_models_dir, "nnUNet", args["configuration"]
    )
    base_nnunet_dir_on_medical = "/media/medical/projects/head_and_neck/nnUnet"
    csv_name = "results"

    ## checkers for input parameters
    # input task
    existing_tasks = {
        int(i.split("_")[0][-3:]): join(nnUNet_configuration_dir, i)
        for i in os.listdir(nnUNet_configuration_dir)
        if i.startswith("Task") and os.path.isdir(join(nnUNet_configuration_dir, i))
    }
    assert (
        args["task_number"] in existing_tasks.keys()
    ), f"Could not find task num.: {args['task_number']}. Found the following task/directories: {existing_tasks}"

    task_dir = existing_tasks[args["task_number"]]
    # e.g.: '/storage/nnUnet/nnUNet_trained_models/nnUNet/3d_fullres/Task152_onkoi-2019-batch-1-and-2-both-modalities-biggest-20-organs-new'
    task_name = Path(task_dir).name
    # e.g.: task_name = 'Task152_onkoi-2019-batch-1-and-2-both-modalities-biggest-20-organs-new'

    ## checkers for input parameters
    # nnunet trainer class
    trainer_classes_list = [i.split("__")[0] for i in os.listdir(task_dir)]
    assert len(trainer_classes_list) > 0, f"no trainer subfolders found in {task_dir}"
    if args["trainer_class_name"] is None:
        if len(trainer_classes_list) > 1:
            ValueError(
                f"Cannot automatically determine trainer class name, since multiple trainer class folders were found in {task_dir}. \nPlease specfiy exact '--trainer_class_name'"
            )
        else:
            args["trainer_class_name"] = trainer_classes_list[0]

    ## checkers for input parameters
    # nnunet plans list
    # determine which plans version was used, raise error if multiple plans exist
    plans_list = [
        i.split("__")[-1]
        for i in os.listdir(task_dir)
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
        task_dir, args["trainer_classes_and_plans_dir_name"]
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
    with open(join(nnUNet_preprocessed_dir, task_name, "dataset.json"), "r") as fp:
        dataset_json_dict = json.load(fp)
    # create modalities dict
    four_digit_ids = {
        m: str.zfill(str(int(i)), 4) for i, m in dataset_json_dict["modality"].items()
    }

    # get image paths with modality four digit id
    raw_data_dir = join(nnUNet_raw_data_base_dir, "nnUNet_raw_data", task_name)
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
    splits_final_dict["test"] = splits_iterator(
        [Path(i).name[: -len(".nii.gz")] for i in dataset_json_dict["test"]],
        dir_name="Ts",
    )
    if not args['just_test']:
        if args["fold"] != "all":
            with open(
                join(nnUNet_preprocessed_dir, task_name, "splits_final.pkl"), "rb"
            ) as f:
                _dict = pickle.load(f)
            splits_final_dict["train"] = splits_iterator(_dict[int(args["fold"])]["train"])
            splits_final_dict["val"] = splits_iterator(_dict[int(args["fold"])]["val"])
        else:
            splits_final_dict["train"] = splits_iterator(
                [
                    Path(_dict["image"]).name[: -len(".nii.gz")]
                    for _dict in dataset_json_dict["training"]
                ]
            )

    images_source_dirs = [
        join(raw_data_dir, "imagesTs"),
    ]

    if not args['just_test']:
        images_source_dirs.append(join(raw_data_dir, "imagesTr"))

    config_str = f"FOLD-{args['fold']}_TRAINER-{args['trainer_class_name']}_PLANS-{args['plans_name']}_CHK-{args['checkpoint_name']}"
    logging.info(f"settings info: {config_str}")
    if args["out_dir"] is None:
        args["out_dir"] = join(base_nnunet_dir_on_medical, task_name, "results")
    os.makedirs(args["out_dir"], exist_ok=True)

    # prepare temporary dir for predicted segmentations
    base_tmp_dir = f"/tmp/nnunet/predict/{task_name}/{config_str}"
    output_seg_dir = f"{base_tmp_dir}/out"
    if os.path.exists(base_tmp_dir):
        shutil.rmtree(base_tmp_dir)
    os.makedirs(output_seg_dir)

    # prepare directories in case predicted segmentations are to be saved
    if args["save_seg_masks"]:
        pred_seg_out_dir = join(args["out_dir"], config_str)
        out_dirs = {
            "test": join(pred_seg_out_dir, "test"),
        }
        if not args['just_test']:
            out_dirs["train"] = join(pred_seg_out_dir, "train")
            out_dirs["val"] = join(pred_seg_out_dir, "val")
            os.makedirs(out_dirs["train"], exist_ok=True)
            if "val" in splits_final_dict.keys():
                os.makedirs(out_dirs["val"], exist_ok=True)
        os.makedirs(out_dirs["test"], exist_ok=True)


    

    try:
        if args['direct_method']:
            from nnunet.inference.predict import predict_from_folder

            model_folder_name = join(
                nnUNet_configuration_dir,
                trainer_classes_and_plans_dir
            )
            print("using model stored in ", model_folder_name)
            assert isdir(model_folder_name), (
                "model output folder not found. Expected: %s" % model_folder_name
            )

            for in_dir in images_source_dirs:
                print(f'\n\nSTARTING predition for {in_dir}\n\n')
                predict_from_folder(
                    model=model_folder_name,
                    input_folder=in_dir,
                    output_folder=output_seg_dir,
                    folds=[args["fold"]],
                    save_npz=False,
                    num_threads_preprocessing=args["num_threads_preprocessing"],
                    num_threads_nifti_save=args["num_threads_nifti_save"],
                    lowres_segmentations=None,
                    part_id=0,
                    num_parts=1,
                    tta=not args["disable_tta"],
                    overwrite_existing=False,
                    mode=args["mode"],
                    overwrite_all_in_gpu=all_in_gpu,
                    mixed_precision=not False,
                    step_size=args["step_size"],
                    checkpoint_name=args["checkpoint_name"],
                )

        else:
            for in_dir in images_source_dirs:
                cmd_list = [
                    "nnUNet_predict",
                    "-i",
                    in_dir,
                    "-o",
                    output_seg_dir,
                    "-t",
                    args["task_number"],
                    "-m",
                    args["configuration"],
                    "-f",
                    args["fold"],
                    "-tr",
                    args["trainer_class_name"],
                    "-chk",
                    args["checkpoint_name"],
                    "--num_threads_preprocessing",
                    args["num_threads_preprocessing"],
                    "--num_threads_nifti_save",
                    args["num_threads_nifti_save"],
                    "--mode",
                    args["mode"],
                    "--disable_tta" if args["disable_tta"] else None,
                ]
                cmd_list = [str(i) for i in cmd_list if i]
                logging.info(f"Final command for nnU-Net prediction: {cmd_list}")

                # set env variables
                if args["gpus"]:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args["gpus"])
                    logging.info(
                        f"Set env variables CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}"
                    )
                os.environ["MKL_THREADING_LAYER"] = "GNU"

                # RUN command in terminal
                subprocess_out = subprocess.run(cmd_list, check=True)

                logging.info(f"Subprocess exit code was: {subprocess_out.returncode}")
                logging.info(f"Successfully predicted seg masks from input dir: {in_dir}")

    except Exception as e:
        logging.error(f"Failed due to the following error: {e}")
        shutil.rmtree(output_seg_dir)
        sys.exit()

    try:
        organs_labels_dict = {
            organ: int(lbl) for lbl, organ in dataset_json_dict["labels"].items()
        }
        logging.info(f"Found the following organs and labels: {organs_labels_dict}")
        compute_metrics = compute_metrics_deepmind(
            organs_labels_dict=organs_labels_dict
        )
        settings_info = {
            "model_task_number": args["task_number"],
            "model_task_name": task_name,
            "fold": args["fold"],
            "trainer_class": args["trainer_class_name"],
            "plans_name": args["plans_name"],
            "checkpoint": args["checkpoint_name"],
            "prediction_mode": args["mode"],
        }
        dfs = []

        for phase, phase_dict in tqdm(
            splits_final_dict.items(), total=len(splits_final_dict)
        ):
            settings_info["phase"] = phase
            for fname, fname_dict in tqdm(phase_dict.items(), total=len(phase_dict)):
                settings_info["fname"] = fname

                gt_fpath = fname_dict["label"]
                pred_fpath = join(output_seg_dir, fname + ".nii.gz")

                out_dict_tmp = compute_metrics.execute(
                    fpath_gt=gt_fpath, fpath_pred=pred_fpath
                )
                df = pd.DataFrame.from_dict(out_dict_tmp)
                for k, val in settings_info.items():
                    df[k] = val
                dfs.append(df)

                if args["save_seg_masks"]:
                    shutil.copy2(pred_fpath, join(out_dirs[phase], fname + ".nii.gz"))

        csv_path = join(args["out_dir"], f"{csv_name}.csv")
        if os.path.exists(csv_path):
            logging.info(
                f"Found existing .csv file on location {csv_path}, merging existing and new dataframe"
            )
            existing_df = [pd.read_csv(csv_path)] + dfs
            pd.concat(existing_df, ignore_index=True).to_csv(csv_path)
        else:
            pd.concat(dfs, ignore_index=True).to_csv(csv_path)
        logging.info(f"Successfully saved {csv_name}.csv file to {csv_path}")
        
    except Exception as e:
        logging.error(f"Failed due to the following error: {e}")

    finally:
        shutil.rmtree(output_seg_dir)
        logging.info(
            f"Successfully deleted temporary pred seg masks dir ({output_seg_dir})"
        )
        sys.exit()


if __name__ == "__main__":
    main()
