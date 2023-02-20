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
import math

from nnunet.training.network_training.nnUNetTrainerV2_DP import nnUNetTrainerV2_DP
from nnunet.training.learning_rate.poly_lr import poly_lr
import numpy as np

class nnUNetTrainerV2_DPnoMirroringAxis2redRot(nnUNetTrainerV2_DP):    
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["mirror_axes"] = (0, 1)
        self.data_aug_params["rotation_x"] = (-90/180*math.pi, 90/180*math.pi)
        print("rotation_x", self.data_aug_params["rotation_x"])
        
    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
            
        if ep >= 1000:
            ep -= 800
            
        self.optimizer.param_groups[0]["lr"] = poly_lr(
            ep, self.max_num_epochs, self.initial_lr, 0.9
        )
        self.print_to_log_file(
            "lr:", np.round(self.optimizer.param_groups[0]["lr"], decimals=6)
        )
