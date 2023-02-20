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


import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast

from nnunet.training.network_training.nnUNet_variants.data_augmentation.nnUNetTrainerV2_DPnoMirroringAxis2redRot import nnUNetTrainerV2_DPnoMirroringAxis2redRot

class nnUNetTrainerV2_DPnoMirroringAxis2redRot_wait200e(nnUNetTrainerV2_DPnoMirroringAxis2redRot):
    def run_iteration(
        self, data_generator, do_backprop=True, run_online_evaluation=False
    ):
        data_dict = next(data_generator)
        data = data_dict["data"]
        target = data_dict["target"]

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        
        if self.epoch < 200:
            data[:,1] *= 0
            print('Multiplied MR image with 0')
            
            # import SimpleITK as sitk
            # for n in range(2):
            #     for c in range(2):
            #         sitk_img = sitk.GetImageFromArray(data[n, c].cpu().numpy())
            #         sitk.WriteImage(sitk_img, f'/media/medical/gasperp/projects/n{n}_c{c}.nii.gz')

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

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

        return l.detach().cpu().numpy()