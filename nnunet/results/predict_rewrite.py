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
import numpy as np
import pandas as pd

sys.path.append(r"/media/medical/gasperp/projects")
sys.path.append(r"/media/medical/gasperp/projects/surface-distance")
from surface_distance import compute_metrics_deepmind



class nnUNet_Prediction_Arg_Parser(object):
    def __init__(self) -> None:
        # Set parser
        self.parser = argparse.ArgumentParser(
            prog="nnU-Net prediction generating script",
            description="Generate & evaluate predictions",
        )
        self.parser.add_argument(
            "-t",
            "--task_number",
            type=int,
            required=True,
            help="Task number of the model used for inference three digit number XXX that comes after TaskXXX_YYYYYY",
        )
        self.parser.add_argument(
            "--dataset_task_number",
            type=int,
            default=None,
            help="Only use this if dataset task number is different that model task number",
        )
        self.parser.add_argument(
            "-f",
            "--fold",
            type=str,
            default=None,
            choices=["0", "1", "2", "3", "4", "all"],
            help="default is None (which means that script automatically determines fold if the is only one fold subfolder, otherwise raises error)",
        )
        self.parser.add_argument(
            "-o",
            "--out_dir",
            type=str,
            default=None,
            help="directory to store output csv file and predictions (if --save_seg_masks is enabled)",
        )
        self.parser.add_argument(
            "--save_seg_masks",
            default=False,
            action="store_true",
            help="if this option is used, output segmentations are stored to out_dir",
        )
        self.parser.add_argument(
            "-conf",
            "--configuration",
            type=str,
            default="3d_fullres",
            choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
            help="nnU-Net configuration",
        )
        self.parser.add_argument(
            "-tr",
            "--trainer_class_name",
            type=str,
            default=None,
            help="nnU-Net trainer: default is None (which means that script automatically determines trainer class if the is only one trainer subfolder, otherwise raises error), common options are nnUNetTrainerV2, nnUNetTrainerV2_noMirroringAxis2",
        )
        self.parser.add_argument(
            "--plans_name",
            type=str,
            default=None,
            help="nnU-Net plans name: default is None (which means that script automatically determines plans name if the is only one trainer subfolder, otherwise raises error), common options are xxx",
        )
        self.parser.add_argument(
            "--checkpoint_name",
            type=str,
            default="model_final_checkpoint",  # this means that 'model_final_checkpoint.model.pkl' is used for inference
            help="nnU-Net model to use: default is final model, but in case inference is done before model training is complete you may want to specifify 'model_best' or 'model_latest'",
        )
        self.parser.add_argument(
            "--phases_to_predict", 
            nargs="+",
            type=str,
            default=['test', 'val', 'train'],
            help="which phases to predict",
        )
        self.parser.add_argument(
            "--gpus",
            nargs="+",
            type=str,
            default=None,
            help="if specified, GPU is utilized to speed up inference",
        )
        self.parser.add_argument(
            "--num_threads_preprocessing",
            type=int,
            default=1,
            help="nnUNet_predict parameter",
        )
        self.parser.add_argument(
            "--num_threads_nifti_save",
            type=int,
            default=1,
            help="nnUNet_predict parameter",
        )
        self.parser.add_argument(
            "--mode", type=str, default="normal", help="nnUNet_predict parameter",
        )
        self.parser.add_argument(
            "--csv_name", type=str, default="results", help="",
        )
        self.parser.add_argument(
            "--mask_modality", type=str, default=None, help="options: CT, MR,... modality",
        )
        self.parser.add_argument(
            "--disable_tta", 
            default=False,
            action="store_true",
            help="nnUNet_predict parameter",
        )
        self.parser.add_argument(
            "--inference_method", 
            default='one-by-one',
            type=str,
            help="method for getting predictions, can be: `one-by-one`, `folder`, `cmd`",
        )
        self.parser.add_argument(
            "--step_size", type=float, default=0.5, required=False, help="don't touch\
                \nstep_size: When running sliding window prediction, the step size determines the distance between adjacent predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between predictions. step_size cannot be larger than 1!"
        )
        self.parser.add_argument(
            "--all_in_gpu",
            type=str,
            default="None",
            required=False,
            help="can be None, False or True",
        )
        
    def checkers(self):
        assert len(self.args['phases_to_predict']) > 0, 'there should be at least one phase'
        
        all_in_gpu = self.args["all_in_gpu"]
        assert all_in_gpu in ["None", "False", "True"]
        if all_in_gpu == "None":
            all_in_gpu = None
        elif all_in_gpu == "True":
            all_in_gpu = True
        elif all_in_gpu == "False":
            all_in_gpu = False
        self.args["all_in_gpu"] = all_in_gpu
        
    def __call__(self):
        # running in terminal
        self.args = vars(self.parser.parse_args())
        
        self.checkers()
        
        predict = Custom_nnUNet_Predict(args=self.args)
        predict.execute_inference()
        
        
    
    
class Custom_nnUNet_Predict(object):
    def __init__(self, args) -> None:
        self.base_nnunet_dir_on_medical = "/media/medical/projects/head_and_neck/nnUnet"
        self.nnUNet_raw_data_base_dir = os.environ["nnUNet_raw_data_base"]
        self.nnUNet_preprocessed_dir = os.environ["nnUNet_preprocessed"]
        self.nnUNet_trained_models_dir = os.environ["RESULTS_FOLDER"]
        
        # fixed prediction parameters for one-by-one
        self.save_npz=False
        self.segs_from_prev_stage=None
        self.mixed_precision=not False
        self.overwrite_existing=False
        self.segmentation_export_kwargs=None
        
        self.results_dir_name = "results"
        
        self.all_in_gpu = args["all_in_gpu"]
        self.configuration = args["configuration"]
        self.task_number = args['task_number']
        
        if args["dataset_task_number"] is None:
            self.dataset_task_number = args["task_number"]
            self.inference_on_unseen_dataset = False
        else:
            self.dataset_task_number=args["dataset_task_number"]
            self.inference_on_unseen_dataset = True

        self.phases_to_predict = args['phases_to_predict']
        self.mode=args["mode"]
        self.gpus=args["gpus"]
        self.step_size = args["step_size"]
        self.do_tta=not args["disable_tta"]
        self.mask_modality = args.get('mask_modality')
        
        self.num_threads_preprocessing=args["num_threads_preprocessing"]
        self.num_threads_nifti_save=args["num_threads_nifti_save"]
        
        # paths definition
        self.nnUNet_configuration_dir = join(
            self.nnUNet_trained_models_dir, "nnUNet", self.configuration
        )
        self.csv_name = args["csv_name"]
        self.save_seg_masks = args["save_seg_masks"]
        
        self.inference_method = args['inference_method']
        self.old_prediciton_method = self.inference_method in ['folder', 'cmd']

        # ------------------------------------------
        ## checkers for input parameters
        # input task
        self.existing_tasks_for_models = {
            int(i.split("_")[0][-3:]): join(self.nnUNet_configuration_dir, i)
            for i in os.listdir(self.nnUNet_configuration_dir)
            if i.startswith("Task") and os.path.isdir(join(self.nnUNet_configuration_dir, i))
        }
        self.existing_tasks_for_datasets = {
            int(i.split("_")[0][-3:]): join(self.base_nnunet_dir_on_medical, i)
            for i in os.listdir(self.base_nnunet_dir_on_medical)
            if i.startswith("Task") and os.path.isdir(join(self.base_nnunet_dir_on_medical, i))
        }
        self.existing_tasks_for_datasets = {**self.existing_tasks_for_models, **self.existing_tasks_for_datasets}
        
        assert (
            self.task_number in self.existing_tasks_for_models.keys()
        ), f"Could not find task num.: {self.task_number}. Found the following task/directories: {self.existing_tasks_for_models}"
        assert (
            self.dataset_task_number in self.existing_tasks_for_datasets.keys()
        ), f"Could not find task num.: {self.dataset_task_number}. Found the following task/directories: {self.existing_tasks_for_datasets}"

        # e.g.: '/storage/nnUnet/nnUNet_trained_models/nnUNet/3d_fullres/TaskXXX_yyyyyy'
        self.model_task_dir = self.existing_tasks_for_models[self.task_number]
        self.dataset_task_dir = self.existing_tasks_for_datasets[self.dataset_task_number]
        
        # e.g.: task_name = 'TaskXXX_yyyyyy'
        self.model_task_name = Path(self.model_task_dir).name
        self.dataset_task_name = Path(self.dataset_task_dir).name

        # nnunet trainer class
        self.get_trainer_class(trainer_class_name=args["trainer_class_name"])

        # nnunet plans list
        self.get_plans_name(plans_name=args['plans_name'])

        # fold
        self.get_fold(fold=args["fold"])

        # ------------------------------------------
        # nnunet model checkpoint to be used for inference
        self.get_model_checkpoint(checkpoint_name=args["checkpoint_name"])

        ## data paths retrieval
        # get dict, dict of dict of filepaths: {'train': {img_name: {'images': {modality0: fpath, ...}, 'label': fpath} ...}, ...}
        # load dataset json
        self.model_dataset_json_dict = self.load_dataset_json(base_dir=self.nnUNet_preprocessed_dir, task_name=self.model_task_name)
        
        self.get_raw_data_dir(task_name=self.dataset_task_name)
        self.dataset_dataset_json_dict = self.load_dataset_json(base_dir=self.raw_data_dir)

        # create modalities dict
        self.four_digit_ids = {
            m: str.zfill(str(int(i)), 4) for i, m in self.model_dataset_json_dict["modality"].items()
        }
        self.MODALITY_to_mask=int(self.four_digit_ids[self.mask_modality]) if self.mask_modality else self.mask_modality

        self.get_final_splits_dict()
        self.get_images_source_dirs()

        # config_str = f"FOLD-{args['fold']}-{args['trainer_class_name']}-{args['plans_name']}_CHK-{args['checkpoint_name']}-{args['dataset_task_number']}_TTA-{not args['disable_tta']}_STEP-{args['step_size']}"
        self.config_str = f"{self.dataset_task_number}_FOLD-{self.fold}_{self.trainer_class_name}_CHK-{self.checkpoint_name}"
        if self.checkpoint_name != 'model_final_checkpoint':
            self.config_str += f"_E-{self.epoch}"
        if self.mask_modality:
            self.config_str += f"_masked-{self.mask_modality}"
            
        logging.info(f"Prediction configuration info: {self.config_str}")
        
        self.get_out_dir(out_dir=args["out_dir"])
            
        # prepare directories in case predicted segmentations are to be saved
        if self.save_seg_masks:
            self.prepare_out_seg_dirs()
            
        if self.old_prediciton_method:
            self.prepare_tmp_dir()
        
    def get_model_epoch(self):
        from nnunet.training.model_restore import load_model_and_checkpoint_files
        trainer, all_params = load_model_and_checkpoint_files(
                self.trainer_class_and_plans_dir, folds=[self.fold], mixed_precision=True, checkpoint_name=self.checkpoint_name
            )
        self.epoch = all_params[0]['epoch']
        del trainer
        del all_params
        logging.info(f'Using model {self.checkpoint_name} checkpoint from epoch {self.epoch}')
        
    def execute_inference(self, inference_method=None, compute_metrics=True):
        # prediction
        if inference_method is None:
            inference_method = self.inference_method
            
        self.successfully_predicted = []
        if inference_method == 'one-by-one':
            self.predict_one_by_one()

        elif inference_method == 'folder':
            self.predict_whole_folder()

        elif inference_method == 'cmd':
            self.predict_cmd()
            
        else:
            raise NotImplementedError('Unknown `--inference_method` was specified')
        
        if compute_metrics:
            self.compute_metrics()
        

    def get_settings_info(self):
        self.settings_info = {
            "model_task_number": self.task_number,
            "model_task_name": self.model_task_name,
            "dataset_task_number": self.dataset_task_number,
            "dataset_task_name": self.dataset_task_name,
            "fold": self.fold,
            "trainer_class": self.trainer_class_name,
            "plans_name": self.plans_name,
            "checkpoint": self.checkpoint_name,
            "prediction_mode": self.mode,
            "TTA": self.do_tta,
            "masked_modality": self.mask_modality
        }
        
    def append_case_specific_info_to_settings_info(self):
        self.settings_info["phase"] = self.case['phase']
        
    def crop_gt_image_to_same_patch_size_as_pred_image(self, gt_fpath, pred_fpath):
        seg_gt = None
        seg_pred = None
        return seg_gt, seg_pred
        
        # compute metrics
    def compute_metrics(self):
        self.organs_labels_dict_pred = {
            organ: int(lbl) for lbl, organ in self.model_dataset_json_dict["labels"].items()
        }
        self.organs_labels_dict_gt = {
            organ: int(lbl) for lbl, organ in self.dataset_dataset_json_dict["labels"].items()
        }
        logging.debug(f"Found the following organs and labels GT dict: {self.organs_labels_dict_gt}")
        logging.debug(f"Found the following organs and labels PRED dict: {self.organs_labels_dict_pred}")
        self.compute_metrics = compute_metrics_deepmind(
            organs_labels_dict_gt=self.organs_labels_dict_gt, organs_labels_dict_pred=self.organs_labels_dict_pred
        )
        self.get_settings_info()
        
        dfs = []
        for self.case in tqdm(
            self.successfully_predicted, total=len(self.successfully_predicted)
        ):
            self.phase = self.case['phase']
            self.gt_fpath = self.case['gt_fpath']
            self.pred_fpath = self.case['pred_fpath']
            self.append_case_specific_info_to_settings_info()
            
            seg_gt, seg_pred = self.crop_gt_image_to_same_patch_size_as_pred_image(gt_fpath=self.gt_fpath, pred_fpath=self.pred_fpath)

            try:
                out_dict_tmp = self.compute_metrics.execute(
                    fpath_gt=self.gt_fpath, fpath_pred=self.pred_fpath, img_gt=seg_gt, img_pred=seg_pred
                )
            except Exception as e:
                logging.error(f"Failed due to the following error: {e}")
                
            df = pd.DataFrame.from_dict(out_dict_tmp)
            for k, val in self.settings_info.items():
                df[k] = val
            dfs.append(df)

            if self.old_prediciton_method and self.save_seg_masks:
                shutil.copy2(self.pred_fpath, join(self.out_dirs[self.phase], self.case['fname'] + ".nii.gz"))

        try:
            csv_path = join(self.out_dir, f"{self.csv_name}.csv")
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

        finally:
            if self.old_prediciton_method:
                shutil.rmtree(self.output_seg_dir)
                logging.info(
                    f"Successfully deleted temporary pred seg masks dir ({self.output_seg_dir})"
                )
            sys.exit()
            
    def get_trainer_class(self, trainer_class_name):
        if trainer_class_name is None:
            trainer_classes_list = [i.split("__")[0] for i in os.listdir(self.model_task_dir)]
            assert len(trainer_classes_list) > 0, f"no trainer subfolders found in {self.model_task_dir}"
            assert len(trainer_classes_list) == 1, ValueError(f"Cannot automatically determine trainer class name, since multiple trainer class folders were found in \n{self.model_task_dir}.\nPlease specfiy exact `--trainer_class_name`")
            self.trainer_class_name = trainer_classes_list[0]
        else:
            self.trainer_class_name = trainer_class_name
        logging.info(f'Using trainer: {self.trainer_class_name}')
            
    def get_plans_name(self, plans_name):
        '''
        determine which plans version was used, raise error if multiple plans exist
        '''
        if plans_name is None:
            plans_list = [
                i.split("__")[-1]
                for i in os.listdir(self.model_task_dir)
                if i.startswith(f"{self.trainer_class_name}__")
            ]
            assert (
                len(plans_list) == 1
            ), ValueError(f"multiple trainer_classes_and_plans dirs found {plans_list}, please specify which to use")
            self.plans_name = plans_list[0]
        else:
            self.plans_name = plans_name
        self.trainer_class_and_plans_dir_name = f"{self.trainer_class_name}__{self.plans_name}"
        self.trainer_class_and_plans_dir = join(
            self.model_task_dir, self.trainer_class_and_plans_dir_name
        )
        assert os.path.isdir(self.trainer_class_and_plans_dir), ValueError(f'Model dir {self.trainer_class_and_plans_dir} does not exist (check `--trainer_class_name` and `--plans_name` options)')
        logging.info(f'Using plans: {self.plans_name}')
        

    
    def get_fold(self, fold):
        get_fold_num = lambda s: int(s.split("_")[-1]) if s != "all" else s
        self.available_folds = [
            i
            for i in os.listdir(self.trainer_class_and_plans_dir)
            if os.path.isdir(join(self.trainer_class_and_plans_dir, i))
        ]
        if "gt_niftis" in self.available_folds:
            self.available_folds.remove("gt_niftis")
        assert (
            len(self.available_folds) > 0
        ), f"no fold subfolders found in {self.trainer_class_and_plans_dir}"

        if fold is None:
            assert len(self.available_folds) == 1, ValueError(
                    f"Cannot automatically determine fold, since multiple folds were found in {self.trainer_class_and_plans_dir}. \nPlease specfiy exact `--fold`"
                )
            fold = get_fold_num(self.available_folds[0])

        if fold != "all":
            # if args['fold'] is 0/1/2/3/4, convert it to 'fold_X' else keep 'all'
            fold = int(fold)
            self.fold_str = f"fold_{fold}"
        else:
            self.fold_str = fold
        
        assert (
            self.fold_str in self.available_folds
        ), f"`--fold` {fold} is not a valid options, available_folds are: {self.available_folds}"
        
        self.fold = fold
        logging.info(f'Using fold {fold}')
    
    def get_model_checkpoint(self, checkpoint_name):
        logging.info("using model stored in ", self.trainer_class_and_plans_dir)
        
        self.models_fold_checkpoints_dir = join(self.trainer_class_and_plans_dir, self.fold_str)
        self.models_checkpoints_dir_files = os.listdir(self.models_fold_checkpoints_dir)
        assert any(
            [checkpoint_name in i for i in self.models_checkpoints_dir_files]
        ), f"--checkpoint_name {checkpoint_name} is not a valid options, checkpoint_name should be a file in {self.models_fold_checkpoints_dir}. Files in this directory are: {self.models_checkpoints_dir_files}"
        self.checkpoint_name = checkpoint_name
        logging.info(f'Using checkpoint {self.checkpoint_name}')
        self.get_model_epoch()

    def load_dataset_json(self, base_dir, task_name=''):
        with open(join(base_dir, task_name, "dataset.json"), "r") as fp:
            return json.load(fp)

    def get_raw_data_dir(self, task_name):
        self.raw_data_dir = join(self.nnUNet_raw_data_base_dir, "nnUNet_raw_data", task_name)
        if not os.path.exists(self.raw_data_dir):
            self.raw_data_dir = join(self.base_nnunet_dir_on_medical, task_name)
        assert os.path.exists(self.raw_data_dir), ValueError(f'raw dataset dir does now exist {self.raw_data_dir}')
        logging.info(f'Using data from {self.raw_data_dir}')
        
    def img_modality_fpaths_dict(self, fname, dir_name): 
        # get image paths with modality four digit id
        return {
            modality: join(self.raw_data_dir, f"images{dir_name}", fname + f"_{m_id}.nii.gz")
            for modality, m_id in self.four_digit_ids.items()
        }
        
    def labels_fpath(self, fname, dir_name): 
        # generate label path
        return join(
            self.raw_data_dir, f"labels{dir_name}", fname + ".nii.gz"
        )
        
    def splits_iterator(self, fname_list, dir_name="Tr"):
        # generate images dict {modality: img_path}
        return {
            img_name: {
                "images": self.img_modality_fpaths_dict(img_name, dir_name=dir_name),
                "label": self.labels_fpath(img_name, dir_name=dir_name),
            }
            for img_name in fname_list
        }
        
    def get_final_splits_dict(self):
        # create dict with paths split on train, val (if fold not 'all') and test
        self.splits_final_dict = {}
        if 'test' in self.phases_to_predict:
            self.splits_final_dict["test"] = self.splits_iterator(
                [Path(i).name[: -len(".nii.gz")] for i in self.model_dataset_json_dict["test"]],
                dir_name="Ts",
            )
        if 'train' in self.phases_to_predict or 'val' in self.phases_to_predict:
            if self.fold == "all" and self.inference_on_unseen_dataset:
                if "val" in self.phases_to_predict.keys():
                    raise UserWarning('there are no images in the validation set (if fold options is `all`, set --phases_to_predict to just `test` `train`)')
                self.splits_final_dict["train"] = self.splits_iterator(
                    [
                        Path(_dict["image"]).name.replace(".nii.gz", '')
                        for _dict in self.model_dataset_json_dict["training"]
                    ]
                )
            else:
                with open(
                    join(self.nnUNet_preprocessed_dir, self.model_task_name, "splits_final.pkl"), "rb"
                ) as f:
                    _dict = pickle.load(f)
                self.splits_final_dict["train"] = self.splits_iterator(_dict[int(self.fold)]["train"])
                self.splits_final_dict["val"] = self.splits_iterator(_dict[int(self.fold)]["val"])

        self.inverse_splits_final_dict  = {cid: phase for phase, cases_dict in self.splits_final_dict.items() for cid, _ in cases_dict.items()}
        logging.info(f'Found data split dicts for the following phases {list(self.splits_final_dict.keys())}')
                
    def get_images_source_dirs(self):
        self.images_source_dirs = []
        if 'test' in self.phases_to_predict:
            self.images_source_dirs.append({'phase': 'test', 'img_dir': join(self.raw_data_dir, "imagesTs"), 'gt_dir': join(self.raw_data_dir, "labelsTs")})
        if 'train' in self.phases_to_predict:
            self.images_source_dirs.append({'phase': 'train', 'img_dir': join(self.raw_data_dir, "imagesTr"), 'gt_dir': join(self.raw_data_dir, "labelsTr")})
        if 'val' in self.phases_to_predict:
            self.images_source_dirs.append({'phase': 'val', 'img_dir': join(self.raw_data_dir, "imagesTr"), 'gt_dir': join(self.raw_data_dir, "labelsTr")})
            
    def get_out_dir(self, out_dir):
        if out_dir is None:
            out_dir = join(self.base_nnunet_dir_on_medical, self.model_task_name, self.results_dir_name)
        elif not out_dir.startswith('/'):
            out_dir = join(self.base_nnunet_dir_on_medical, self.model_task_name, self.results_dir_name, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        logging.info(f'Using output directoty {out_dir}')
    
    def prepare_tmp_dir(self):
        # prepare temporary dir for predicted segmentations
        base_tmp_dir = f"/tmp/nnunet/predict/{self.model_task_name}/{self.config_str}"
        self.output_seg_dir = f"{base_tmp_dir}/out"
        # if os.path.exists(base_tmp_dir):
        #     shutil.rmtree(base_tmp_dir)
        os.makedirs(self.output_seg_dir, exist_ok=True)
        
    def prepare_out_seg_dirs(self):
        pred_seg_out_dir = join(self.out_dir, self.config_str)
        self.out_dirs = {}

        if 'test' in self.splits_final_dict:
            self.out_dirs["test"] = join(pred_seg_out_dir, "test")
            os.makedirs(self.out_dirs["test"], exist_ok=True)
            
        if 'train' in self.splits_final_dict:
            self.out_dirs["train"] = join(pred_seg_out_dir, "train")
            os.makedirs(self.out_dirs["train"], exist_ok=True)
            
        if 'val' in self.splits_final_dict:
            self.out_dirs["val"] = join(pred_seg_out_dir, "val")
            os.makedirs(self.out_dirs["val"], exist_ok=True)

    
    
    def iterate_image_source_dirs(self):
        for _dict_tmp in self.images_source_dirs:
            phase = _dict_tmp['phase']
            img_dir = _dict_tmp['img_dir']
            logging.info(f'Starting prediction for {img_dir}')
            yield phase, img_dir, _dict_tmp
        
    
    def predict_one_by_one(self):
        from nnunet.inference.predict import check_input_folder_and_return_caseIDs, subfiles, load_pickle
        
        assert self.mode == 'normal', NotImplementedError('current implementation for one-by-one_method supports only normal mode')
        
        self.expected_num_modalities = load_pickle(join(self.trainer_class_and_plans_dir, "plans.pkl"))["num_modalities"]

        for self.phase, self.img_dir, _dict_tmp in self.iterate_image_source_dirs():
            case_ids = check_input_folder_and_return_caseIDs(
                self.img_dir, self.expected_num_modalities
            )
            case_ids = [cid for cid in case_ids if self.inverse_splits_final_dict[cid] == self.phase]
            
            print(f'\n\nFound {len(case_ids)} cases in phase {self.phase}\n\n')
            
            output_files = [join(self.out_dirs[self.phase], cid + ".nii.gz") for cid in case_ids]
            gt_files = [join(_dict_tmp['gt_dir'], cid + ".nii.gz") for cid in case_ids]
            all_files = subfiles(self.img_dir, suffix=".nii.gz", join=False, sort=True)
            list_of_lists = [
                [
                    join(self.img_dir, i)
                    for i in all_files
                    if i[: len(j)].startswith(j) and len(i) == (len(j) + 12)
                ]
                for j in case_ids
            ]

            for self.input_filename, self.output_filename, self.gt_filename in zip(list_of_lists, output_files, gt_files):
                self.predict_single_case()
                    
    def predict_single_case(self):
        from nnunet.inference.predict import predict_cases
        import torch
        try:
            torch.cuda.empty_cache()
            if not os.path.exists(self.output_filename):
                predict_cases(
                    model=self.trainer_class_and_plans_dir,
                    list_of_lists=[self.input_filename],
                    output_filenames=[self.output_filename],
                    folds=[self.fold],
                    save_npz=self.save_npz,
                    num_threads_preprocessing=self.num_threads_preprocessing,
                    num_threads_nifti_save=self.num_threads_nifti_save,
                    segs_from_prev_stage=self.segs_from_prev_stage,
                    do_tta=self.do_tta,
                    mixed_precision=self.mixed_precision,
                    overwrite_existing=self.overwrite_existing,
                    all_in_gpu=self.all_in_gpu,
                    step_size=self.step_size,
                    checkpoint_name=self.checkpoint_name,
                    segmentation_export_kwargs=self.segmentation_export_kwargs,
                    MODALITY_to_mask=self.MODALITY_to_mask
                )
            else:
                logging.info('Skipping inference, because output file exists')
            self.append_to_successful_list(
                fname=Path(self.input_filename[0]).name.replace('.nii.gz', ''), 
                pred_fpath=self.output_filename, 
                gt_fpath=self.gt_filename
            )
        except Exception as e:
            logging.error(f"Failed due to predict case {self.input_filename} due to the following error: {e}")
                    
    def append_to_successful_list(self, fname, pred_fpath, gt_fpath):
        self.successfully_predicted.append({'fname': fname, 
                            'pred_fpath': pred_fpath,
                            'gt_fpath': gt_fpath,
                            'phase': self.phase})
        
    def list_output_seg_dir_and_append_to_successful_list(self):
        for fn in os.listdir(self.output_seg_dir):
            self.append_to_successful_list(
                    fname=fn.replace('.nii.gz', ''),
                    pred_fpath=join(self.output_seg_dir, fn), 
                    gt_fpath=join(self._dict_tmp['gt_dir'], fn)
                )
            
    def predict_patch(self, d, dct):
        if self.MODALITY_to_mask is not None:
            d[self.MODALITY_to_mask] = 0
            print(f'MASKING modality with index {self.MODALITY_to_mask}')

        print("predicting", self.output_filename)
        self.trainer.load_checkpoint_ram(self.params[0], False)
        softmax = self.trainer.predict_preprocessed_data_return_seg_and_softmax(
            d,
            do_mirroring=False,
            mirror_axes=[],
            use_sliding_window=False,
            all_in_gpu=self.all_in_gpu,
            mixed_precision=self.mixed_precision,
        )[1]


        transpose_forward = self.trainer.plans.get("transpose_forward")
        if transpose_forward is not None:
            transpose_backward = self.trainer.plans.get("transpose_backward")
            softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

        if self.save_npz:
            npz_file = self.output_filename[:-7] + ".npz"
        else:
            npz_file = None

        if hasattr(self.trainer, "regions_class_order"):
            region_class_order = self.trainer.regions_class_order
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
        if self.all_in_gpu:
            bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
        if np.prod(softmax.shape) > (
            2e9 / bytes_per_voxel * 0.85
        ):  # * 0.85 just to be save
            print(
                "This output is too large for python process-process communication. Saving output temporarily to disk"
            )
            np.save(self.output_filename[:-7] + ".npy", softmax)
            softmax = self.output_filename[:-7] + ".npy"

        self.results.append(
            self.pool.starmap_async(
                self.save_segmentation_nifti_from_softmax,
                (
                    (
                        softmax,
                        self.output_filename,
                        dct,
                        self.interpolation_order,
                        region_class_order,
                        None,
                        None,
                        npz_file,
                        None,
                        self.force_separate_z,
                        self.interpolation_order_z,
                    ),
                ),
            )
        )
        self.append_to_successful_list(
            fname=Path(self.input_filename[0]).name.replace('.nii.gz', ''), 
            pred_fpath=self.output_filename, 
            gt_fpath=self.gt_filename
        )
        
        
    def predict_whole_folder(self):
        from nnunet.inference.predict import predict_from_folder
        for self.phase, self.img_dir, self._dict_tmp in self.iterate_image_source_dirs():
            predict_from_folder(
                model=self.trainer_class_and_plans_dir,
                input_folder=self.img_dir,
                output_folder=self.output_seg_dir,
                folds=[self.fold],
                save_npz=False,
                num_threads_preprocessing=self.num_threads_preprocessing,
                num_threads_nifti_save=self.num_threads_nifti_save,
                lowres_segmentations=None,
                part_id=0,
                num_parts=1,
                tta=self.do_tta,
                overwrite_existing=False,
                mode=self.mode,
                overwrite_all_in_gpu=self.all_in_gpu,
                mixed_precision=not False,
                step_size=self.step_size,
                checkpoint_name=self.checkpoint_name,
            )
            
        # TODO: needs a check
        self.list_output_seg_dir_and_append_to_successful_list()
            
                
    def predict_cmd(self):
        try:
            for self.phase, self.img_dir, self._dict_tmp in self.iterate_image_source_dirs():
                cmd_list = [
                    "nnUNet_predict",
                    "-i",
                    self.img_dir,
                    "-o",
                    self.output_seg_dir,
                    "-t",
                    self.task_number,
                    "-m",
                    self.configuration,
                    "-f",
                    self.fold,
                    "-tr",
                    self.trainer_class_name,
                    "-chk",
                    self.checkpoint_name,
                    "--num_threads_preprocessing",
                    self.num_threads_preprocessing,
                    "--num_threads_nifti_save",
                    self.num_threads_nifti_save,
                    "--mode",
                    self.mode,
                    "--disable_tta" if not self.do_tta else None,
                ]
                cmd_list = [str(i) for i in cmd_list if i]
                logging.info(f"Final command for nnU-Net prediction: {cmd_list}")

                # set env variables
                if self.gpus is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpus)
                    logging.info(
                        f"Set env variables CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}"
                    )
                os.environ["MKL_THREADING_LAYER"] = "GNU"

                # RUN command in terminal
                subprocess_out = subprocess.run(cmd_list, check=True)

                logging.info(f"Subprocess exit code was: {subprocess_out.returncode}")
                logging.info(f"Successfully predicted seg masks from input dir: {self.img_dir}")
                
            # TODO: needs a check
            self.list_output_seg_dir_and_append_to_successful_list()
        except Exception as e:
            logging.error(f"Failed due to the following error: {e}")
            shutil.rmtree(self.output_seg_dir)
            sys.exit()

if __name__ == "__main__":
    run = nnUNet_Prediction_Arg_Parser()
    run()
