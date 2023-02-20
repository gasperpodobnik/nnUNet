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


from batchgenerators.utilities.file_and_folder_operations import *
import torch
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast

from nnunet.training.network_training.nnUNet_variants.data_augmentation.nnUNetTrainerV2_DPnoMirroringAxis2redRot import nnUNetTrainerV2_DPnoMirroringAxis2redRot

class nnUNetTrainerV2_DP_singleModal0(nnUNetTrainerV2_DPnoMirroringAxis2redRot):
    def process_plans(self, plans):
        super().process_plans(plans)
        self.num_input_channels = 1
        self.print_to_log_file('Using only one modality, modality with index 0')
        
    def run_iteration(
        self, data_generator, do_backprop=True, run_online_evaluation=False
    ):
        data_dict = next(data_generator)
        data = data_dict["data"]
        target = data_dict["target"]

        data = data[:,:1]
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()
        
        # import SimpleITK as sitk
        # rnd_run_id = np.random.randint(1000)
        # for b in range(data.shape[0]):
        #     for m in range(data.shape[1]):
        #         sitk_img = sitk.GetImageFromArray(data[b, m].cpu().numpy())
        #         sitk_img.SetSpacing(data_dict['properties'][b]['spacing_after_resampling'][::-1].tolist())
        #         sitk.WriteImage(sitk_img, os.path.join('/media/medical/projects/head_and_neck/nnUnet/Task107_CTAV/training_examples' +  f'{rnd_run_id}_{b}_{m}.nii.gz'))

        if self.fp16:
            with autocast():
                ret = self.network(
                    data, target, return_hard_tp_fp_fn=run_online_evaluation
                )
                if run_online_evaluation:
                    ces, tps, fps, fns, tp_hard, fp_hard, fn_hard = ret
                    self.run_online_evaluation(tp_hard, fp_hard, fn_hard)
                else:
                    ces, tps, fps, fns = ret
                del data, target
                l = self.compute_loss(ces, tps, fps, fns)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            ret = self.network(data, target, return_hard_tp_fp_fn=run_online_evaluation)
            if run_online_evaluation:
                ces, tps, fps, fns, tp_hard, fp_hard, fn_hard = ret
                self.run_online_evaluation(tp_hard, fp_hard, fn_hard)
            else:
                ces, tps, fps, fns = ret
            del data, target
            l = self.compute_loss(ces, tps, fps, fns)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        # import nnunet.results.fcn as fcn
        # fcn.plot_grad_flow(self.network.named_parameters(), str(self.epoch) + '_' +str(rnd_run_id))
        return l.detach().cpu().numpy()