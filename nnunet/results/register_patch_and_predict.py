import logging
import os
import copy
from os.path import join
from pathlib import Path
import nnunet.results.fcn as fcn
import nnunet.results.predict as predict
from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.model_restore import load_model_and_checkpoint_files
import numpy as np
import torch
import SimpleITK as sitk
from nnunet.inference.predict import preprocess_multithreaded


class nnUNet_Registration_Arg_Parser(predict.nnUNet_Prediction_Arg_Parser):
    def __init__(self) -> None:
        super().__init__()
        self.parser.add_argument(
            "--organs_of_interest", 
            nargs="+",
            type=str,
            required=True,
            help="organ names for which registration will be performed",
        )
        self.parser.add_argument(
            "--deformable",
            default=False,
            action="store_true",
            help="if this option is used, deformable reg. will be performed after rigid reg.",
        )
    def checkers(self):
        super().checkers()
        assert self.args['inference_method'] == 'one-by-one', NotImplementedError('patch registration currently only works with `one-by-one` inference method')
        
    def __call__(self):
        # running in terminal
        self.args = vars(self.parser.parse_args())
        
        self.checkers()
        predict = Patch_Registration_nnUNet_Predict(args=self.args)
        predict.execute_inference()
        


class Patch_Registration_nnUNet_Predict(predict.Custom_nnUNet_Predict):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.rigid = True
        self.deformable = args['deformable']
        self.organs_of_interest = args['organs_of_interest']
        self.results_dir_name = "patch_registration_results"
        logging.info(f'Started `Patch_Registration_nnUNet_Predict`')
        
        
        
    def execute_inference(self, compute_metrics=True):
        self.prepare_dirs_and_outputs()
        
        self.predict_one_by_one()
        
        if compute_metrics:
            self.compute_metrics()
        
    def predict_single_case(self):
        self.reg_patch_and_predict()
        
        
    def prepare_preprocessing_stuff(self):
        self.list_of_lists=[self.input_filename]
        self.output_filenames=[self.output_filename]
        self.folds=[self.fold]
        
        assert len(self.list_of_lists) == len(self.output_filenames)
        
        if self.segs_from_prev_stage is not None:
            assert len(self.segs_from_prev_stage) == len(self.output_filenames)

        self.pool = Pool(self.num_threads_nifti_save)
        self.results = []

        cleaned_output_files = []
        for o in self.output_filenames:
            dr, f = os.path.split(o)
            if len(dr) > 0:
                maybe_mkdir_p(dr)
            if not f.endswith(".nii.gz"):
                f, _ = os.path.splitext(f)
                f = f + ".nii.gz"
            cleaned_output_files.append(join(dr, f))

        if not self.overwrite_existing:
            print("number of cases:", len(self.list_of_lists))
            # if save_npz=True then we should also check for missing npz files
            not_done_idx = [
                i
                for i, j in enumerate(cleaned_output_files)
                if (not isfile(j)) or (self.save_npz and not isfile(j[:-7] + ".npz"))
            ]

            cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
            self.list_of_lists = [self.list_of_lists[i] for i in not_done_idx]
            if self.segs_from_prev_stage is not None:
                self.segs_from_prev_stage = [self.segs_from_prev_stage[i] for i in not_done_idx]

            print(
                "number of cases that still need to be predicted:",
                len(cleaned_output_files),
            )

        print("emptying cuda cache")
        torch.cuda.empty_cache()

        print("loading parameters for folds,", self.folds)
        self.trainer, self.params = load_model_and_checkpoint_files(
            self.trainer_class_and_plans_dir, self.folds, mixed_precision=self.mixed_precision, checkpoint_name=self.checkpoint_name
        )
        
        if len(self.params) > 1:
            raise NotImplementedError('num params greater than 1, this is not implemented yet, see original nnunet code')

        if self.segmentation_export_kwargs is None:
            if "segmentation_export_params" in self.trainer.plans.keys():
                self.force_separate_z = self.trainer.plans["segmentation_export_params"][
                    "force_separate_z"
                ]
                self.interpolation_order = self.trainer.plans["segmentation_export_params"][
                    "interpolation_order"
                ]
                self.interpolation_order_z = self.trainer.plans["segmentation_export_params"][
                    "interpolation_order_z"
                ]
            else:
                self.force_separate_z = None
                self.interpolation_order = 1
                self.interpolation_order_z = 0
        else:
            self.force_separate_z = self.segmentation_export_kwargs["force_separate_z"]
            self.interpolation_order = self.segmentation_export_kwargs["interpolation_order"]
            self.interpolation_order_z = self.segmentation_export_kwargs["interpolation_order_z"]
            
        print("starting preprocessing generator")
        self.preprocessing = preprocess_multithreaded(
            self.trainer,
            self.list_of_lists,
            cleaned_output_files,
            self.num_threads_preprocessing,
            self.segs_from_prev_stage,
        )
        self.organs_labels_dict = {j: int(i) for i, j in self.dataset_dataset_json_dict['labels'].items()}
    
    def reg_patch_and_predict(self):
        self.prepare_preprocessing_stuff()
        print("starting prediction...")
        all_output_files = []
        for preprocessed in self.preprocessing:
            output_filename, (d, dct) = preprocessed
            all_output_files.append(all_output_files)
            logging.info(f"Registering case: {Path(output_filename).name}")
            if isinstance(d, str):
                data = np.load(d)
                os.remove(d)
                d = data
            for self.organ_of_interest in self.organs_of_interest:
                logging.info(f'Registering {self.organ_of_interest}')
                self.organ_lbl_int = self.organs_labels_dict[self.organ_of_interest]
                try:
                    for enum, _d, _dct in self.register_patch(d, dct):
                        self.predict_patch(_d, _dct)
                except Exception as e:
                    logging.error(f"Failed due to the following error: {e}")
            
        print("inference done. Now waiting for the segmentation export to finish...")
        _ = [i.get() for i in self.results]
        # now apply postprocessing
        # first load the postprocessing properties if they are present. Else raise a well visible warning
        self.pool.close()
        self.pool.join()
                
    def append_to_successful_list(self, fname, pred_fpath, gt_fpath):
        self.successfully_predicted.append({
            'fname': fname, 
            'pred_fpath': pred_fpath,
            'gt_fpath': gt_fpath,
            'phase': self.phase, 
            'rigid': self.isRigid, 
            'deformable': self.isDeformable,
            'organ_of_interest': self.organ_of_interest
            }
        )
            
    def register_patch(self, d, dct):
        final_patch_size = np.array([192, 192, 40])
        rough_patch_size = final_patch_size + np.array([20, 20, 6])
        
        d = np.copy(d)
        dct = copy.deepcopy(dct)
        
        organ_results_dir = join(self.out_dir, self.organ_of_interest)
        os.makedirs(organ_results_dir, exist_ok=True)
        
        fname_id = Path(self.gt_filename).name.split('_')[-1].split('.')[0] # three digit number, e.g. `001`
        seg_sitk = sitk.ReadImage(self.gt_filename)
        
        assert (sitk.GetArrayFromImage(seg_sitk) == self.organ_lbl_int).sum() > 0, f'{self.organ_of_interest} is not in GT seg'
        
        default_pixel_value_ct = float(d[0].min())
        default_pixel_value_mr = float(d[1].min())
        ct_sitk = sitk.GetImageFromArray(d[0])
        ct_sitk.CopyInformation(seg_sitk)
        mr_sitk = sitk.GetImageFromArray(d[1])
        mr_sitk.CopyInformation(seg_sitk)
        
        
        # compute organ bbox
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(seg_sitk)
        bbox_idx = np.array(label_shape_filter.GetBoundingBox(self.organ_lbl_int))
        bbox_start_idx, bbox_size = bbox_idx[:3], bbox_idx[3:]
        bbox_end_idx = bbox_start_idx + bbox_size - 1
        bbox_center_idx = (bbox_start_idx + bbox_end_idx)/2
        

        # --------------------------------------------------------------------------------
        # extract small patch around bbox of the organ of interest and register it rigidly
        bbox_extension = (bbox_size/4).astype(int)
        minimal_extension = (10/np.array(seg_sitk.GetSpacing())).astype(int)
        bbox_extension[bbox_extension < minimal_extension] = minimal_extension[bbox_extension < minimal_extension]
        reg_patch_size = bbox_size + 2*bbox_extension
        reg_patch_start_idx = bbox_start_idx - bbox_extension
        
        assert np.all(reg_patch_size >= 4), f'Patch around {self.organ_of_interest} is too smal'
        
        reg_resample_filter = sitk.ResampleImageFilter()
        reg_resample_filter.SetReferenceImage(seg_sitk)
        reg_resample_filter.SetOutputOrigin(seg_sitk.TransformIndexToPhysicalPoint(reg_patch_start_idx.tolist()))
        reg_resample_filter.SetSize(reg_patch_size.tolist())
        reg_resample_filter.SetInterpolator(sitk.sitkBSpline)
        reg_resample_filter.SetDefaultPixelValue(default_pixel_value_ct)
        ct_rigid_reg_patch_sitk = reg_resample_filter.Execute(ct_sitk)
        reg_resample_filter.SetDefaultPixelValue(default_pixel_value_mr)
        mr_rigid_reg_patch_sitk = reg_resample_filter.Execute(mr_sitk)
        
        rigid_transform = fcn.register_patches(fixed_patch=ct_rigid_reg_patch_sitk, moving_patch=mr_rigid_reg_patch_sitk, rigid=self.rigid, deformable=False, root_dir=organ_results_dir)
        
        rigid_reg_mr_sitk = fcn.apply_transform(mr_sitk, rigid_transform)
        reg_mr_sitk = rigid_reg_mr_sitk
        # --------------------------------------------------------------------------------
        
        if self.deformable:
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
            
            deformable_transform = fcn.register_patches(fixed_patch=ct_deformable_reg_patch_sitk, moving_patch=mr_deformable_reg_patch_sitk, rigid=False, deformable=True, root_dir=organ_results_dir)
            
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



        fp_root = join(organ_results_dir, fname_id + '_')
        for ee, t in enumerate(rigid_transform):
            sitk.WriteParameterFile(t, fp_root + str(ee) + '_tp_map_rigid.txt')
        if self.deformable:
            for ee, t in enumerate(deformable_transform):
                sitk.WriteParameterFile(t, fp_root + str(ee) + '_tp_map_deformable.txt')
        sitk.WriteImage(ct_final_patch_sitk, fp_root + 'ct_final_patch_sitk.nii.gz')
        sitk.WriteImage(unreg_mr_final_patch_sitk, fp_root + 'unreg_mr_final_patch_sitk.nii.gz')
        sitk.WriteImage(reg_mr_final_patch_sitk, fp_root + 'reg_mr_final_patch_sitk.nii.gz')
        sitk.WriteImage(rigid_reg_mr_final_patch_sitk, fp_root + 'rigid_reg_mr_final_patch_sitk.nii.gz')
        sitk.WriteImage(sitk.Cast(seg_final_patch_sitk==self.organ_lbl_int, sitk.sitkUInt8), fp_root + 'seg_final_patch_sitk.nii.gz')

            
        # translate_filter = sitk.ResampleImageFilter()
        # translate_filter.SetReferenceImage(seg_sitk)
        # translate_filter.SetInterpolator(sitk.sitkBSpline)
        # translate_filter.SetDefaultPixelValue(default_pixel_value_mr)
        
        # ct_patch = sitk.GetArrayFromImage(ct_final_patch_sitk)
        dct['crop_bbox'] = None #[[i, j] for i, j in zip(real_start[::-1], real_end[::-1])]
        dct['size_after_cropping'] = final_patch_size[::-1]
        dct['itk_origin'] = ct_final_patch_sitk.GetOrigin()
        
        to_predict = [[unreg_mr_final_patch_sitk, False, False], [rigid_reg_mr_final_patch_sitk, self.rigid, False]]
        if self.deformable:
            to_predict.append([reg_mr_final_patch_sitk, self.rigid, self.deformable])
            
            
        # output_filename_orig = output_filename
        # for distance in distances:
        # fcn.transform_and_get_patch(distance, N, mr_sitk, rough_resample_filter, roi_filter)
        for enum, (mr_patch, self.isRigid, self.isDeformable) in enumerate(to_predict):
            # print(distance, translation_params)
            # output_filename = output_filename_orig.replace('.nii.gz', f'_D{distance}_{enum}.nii.gz')
            self.output_filename = fp_root + f'_RED-SEG_rigid-{self.isRigid}_deformable-{self.isDeformable}.nii.gz'
            d = np.stack((sitk.GetArrayFromImage(ct_final_patch_sitk), sitk.GetArrayFromImage(mr_patch)))
            yield enum, d, dct
        
    def append_case_specific_info_to_settings_info(self):
        super().append_case_specific_info_to_settings_info()
        self.settings_info['organ_of_interest'] = self.case['organ_of_interest']
        self.settings_info['rigid']=self.case.get('rigid')
        self.settings_info['deformable']=self.case.get('deformable')
        # self.settings_info['distance']=self.case.get('distance')
        # self.settings_info['translation_params']=str(self.case.get('translation_params'))
        # self.settings_info['patch_enum']=self.case.get('enum')
        # self.settings_info['reged_organ']=self.case.get('reged_organ')
        
    def crop_gt_image_to_same_patch_size_as_pred_image(self, gt_fpath, pred_fpath):
        seg_pred = sitk.ReadImage(pred_fpath)
        seg_gt = sitk.ReadImage(gt_fpath)
        
        # crop gt image to same patch size as pred image
        final_resample_filter = sitk.ResampleImageFilter()
        final_resample_filter.SetReferenceImage(seg_pred)
        final_resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
        final_resample_filter.SetDefaultPixelValue(0)
        seg_gt = final_resample_filter.Execute(seg_gt)
        return seg_gt, seg_pred


if __name__ == "__main__":
    run = nnUNet_Registration_Arg_Parser()
    run()
