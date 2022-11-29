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

import pandas as pd

sys.path.append(r"/media/medical/gasperp/projects")
sys.path.append(r"/media/medical/gasperp/projects/surface-distance")
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
        help="nnU-Net model to use: default is final model, but in case inference is done before model training is complete you may want to specifify 'model_best.model.pkl' or sth else",
    )
    parser.add_argument(
        "--phases_to_predict", 
        nargs="+",
        type=str,
        default=['test', 'train'],
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
    csv_name = "results"

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
    if 'train' in args['phases_to_predict']:
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

    if 'train' in args['phases_to_predict']:
        images_source_dirs.append({'phase': 'train', 'img_dir': join(raw_data_dir, "imagesTr"), 'gt_dir': join(raw_data_dir, "labelsTr")})

    config_str = f"FOLD-{args['fold']}-{args['trainer_class_name']}-{args['plans_name']}_CHK-{args['checkpoint_name']}-{args['dataset_task_number']}_TTA-{not args['disable_tta']}_STEP-{args['step_size']}"
    logging.info(f"settings info: {config_str}")
    if args["out_dir"] is None:
        args["out_dir"] = join(base_nnunet_dir_on_medical, model_task_name, "results")
    os.makedirs(args["out_dir"], exist_ok=True)

    # prepare temporary dir for predicted segmentations
    base_tmp_dir = f"/tmp/nnunet/predict/{model_task_name}/{config_str}"
    output_seg_dir = f"{base_tmp_dir}/out"
    # if os.path.exists(base_tmp_dir):
    #     shutil.rmtree(base_tmp_dir)
    os.makedirs(output_seg_dir, exist_ok=True)

    # prepare directories in case predicted segmentations are to be saved
    if args["save_seg_masks"]:
        pred_seg_out_dir = join(args["out_dir"], config_str)
        out_dirs = {}

        if 'test' in args['phases_to_predict']:
            out_dirs["test"] = join(pred_seg_out_dir, "test")
            os.makedirs(out_dirs["test"], exist_ok=True)
        if 'train' in args['phases_to_predict']:
            out_dirs["train"] = join(pred_seg_out_dir, "train")
            out_dirs["val"] = join(pred_seg_out_dir, "val")
            os.makedirs(out_dirs["train"], exist_ok=True)
            if "val" in splits_final_dict.keys():
                os.makedirs(out_dirs["val"], exist_ok=True)


    

    model_folder_name = join(
                nnUNet_configuration_dir,
                trainer_classes_and_plans_dir
            )
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), (
        "model output folder not found. Expected: %s" % model_folder_name
    )
    
    successfully_predicted = []
    try:
        if args['inference_method'] == 'one-by-one':
            assert args["mode"] == 'normal', NotImplementedError('current implementation for one-by-one_method supports only normal mode')
            from nnunet.inference.predict import predict_cases, check_input_folder_and_return_caseIDs, subfiles, load_pickle
            import torch

            expected_num_modalities = load_pickle(join(model_folder_name, "plans.pkl"))["num_modalities"]
            for _dict_tmp in images_source_dirs:
                phase = _dict_tmp['phase']
                img_dir = _dict_tmp['img_dir']
                case_ids = check_input_folder_and_return_caseIDs(
                    img_dir, expected_num_modalities
                )
                # case_ids = case_ids[:28]
                output_files = [join(out_dirs[phase], i + ".nii.gz") for i in case_ids]
                gt_files = [join(_dict_tmp['gt_dir'], i + ".nii.gz") for i in case_ids]
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
                            predict_cases(
                                model=model_folder_name,
                                list_of_lists=[input_filename],
                                output_filenames=[output_filename],
                                folds=[args["fold"]],
                                save_npz=False,
                                num_threads_preprocessing=args["num_threads_preprocessing"],
                                num_threads_nifti_save=args["num_threads_nifti_save"],
                                segs_from_prev_stage=None,
                                do_tta=do_tta,
                                mixed_precision=not False,
                                overwrite_existing=False,
                                all_in_gpu=all_in_gpu,
                                step_size=step_size,
                                checkpoint_name=args["checkpoint_name"],
                                segmentation_export_kwargs=None,
                            )
                        else:
                            logging.info('Skipping inference, because output file exists')
                        successfully_predicted.append({'fname': Path(input_filename[0]).name.replace('.nii.gz', ''), 
                                                       'pred_fpath': output_filename, 
                                                       'gt_fpath': gt_filename, 
                                                       'phase': phase})
                    except Exception as e:
                        logging.error(f"Failed due to predict case {input_filename} due to the following error: {e}")
        elif args['inference_method'] == 'folder':
            # TODO implement saving results paths to `successfully_predicted`
            from nnunet.inference.predict import predict_from_folder
            for _dict_tmp in images_source_dirs:
                phase = _dict_tmp['phase']
                img_dir = _dict_tmp['img_dir']
                print(f'\n\nSTARTING predition for {img_dir}\n\n')
                predict_from_folder(
                    model=model_folder_name,
                    input_folder=img_dir,
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
                for fn in os.listdir(output_seg_dir):
                    successfully_predicted.append({'fname': fn.replace('.nii.gz', ''), 
                                                        'pred_fpath': join(output_seg_dir, fn), 
                                                        'gt_fpath': join(_dict_tmp['gt_dir'], fn), 
                                                        'phase': phase})

        elif args['inference_method'] == 'cmd':
            # TODO implement saving results paths to `successfully_predicted`
            for _dict_tmp in images_source_dirs:
                phase = _dict_tmp['phase']
                img_dir = _dict_tmp['img_dir']
                cmd_list = [
                    "nnUNet_predict",
                    "-i",
                    img_dir,
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
                logging.info(f"Successfully predicted seg masks from input dir: {img_dir}")
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
        }
        dfs = []

        for case in tqdm(
            successfully_predicted, total=len(successfully_predicted)
        ):
            settings_info["phase"] = case['phase']
            # settings_info["fname"] = case['fname']

            gt_fpath = case['gt_fpath']
            pred_fpath = case['pred_fpath']

            out_dict_tmp = compute_metrics.execute(
                fpath_gt=gt_fpath, fpath_pred=pred_fpath
            )
            df = pd.DataFrame.from_dict(out_dict_tmp)
            for k, val in settings_info.items():
                df[k] = val
            dfs.append(df)

            # if args["save_seg_masks"]:
            #     shutil.copy2(pred_fpath, join(out_dirs[phase], fname + ".nii.gz"))

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
        shutil.rmtree(output_seg_dir)
        logging.info(
            f"Successfully deleted temporary pred seg masks dir ({output_seg_dir})"
        )
        sys.exit()


if __name__ == "__main__":
    main()
