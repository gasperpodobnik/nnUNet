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

from pathlib import Path
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_DPnoMirroringAxis2redRot_dualPath_sep2 import nnUNetTrainerV2_DPnoMirroringAxis2redRot_dualPath_sep2


class nnUNetTrainerV2_DPnoMirroringAxis2redRot_dualPath_sep2_blockModal0_freezeCT_normalTraining_weightedLoss(nnUNetTrainerV2_DPnoMirroringAxis2redRot_dualPath_sep2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None, unpack_data=True, deterministic=True, num_gpus=1, distribute_batch_size=False, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, num_gpus, distribute_batch_size, fp16)

        self.organ_weights_dict = {'A_Carotid_L': 0.8397093548684759,
        'A_Carotid_R': 0.8524652371969953,
        'Arytenoid': 0.544266449546219,
        'Bone_Mandible': 0.9442748766745901,
        'Brainstem': 0.8664332202306968,
        'BuccalMucosa': 0.6871962096220007,
        'Cavity_Oral': 0.9000504798769675,
        'Cochlea_L': 0.7446893642765099,
        'Cochlea_R': 0.7293121642171246,
        'Cricopharyngeus': 0.6551170466718486,
        'Esophagus_S': 0.6573392798298432,
        'Eye_AL': 0.8161203233855117,
        'Eye_AR': 0.8116755652395643,
        'Eye_PL': 0.9306403115450865,
        'Eye_PR': 0.9185019773584973,
        'Glnd_Lacrimal_L': 0.6573003702975765,
        'Glnd_Lacrimal_R': 0.6199768113209791,
        'Glnd_Submand_L': 0.8418086358924579,
        'Glnd_Submand_R': 0.8420031305472915,
        'Glnd_Thyroid': 0.9082470474770261,
        'Glottis': 0.7359847253961697,
        'Larynx_SG': 0.8085505433322624,
        'Lips': 0.7267610842278195,
        'OpticChiasm': 0.4547367902314752,
        'OpticNrv_L': 0.6951258569096138,
        'OpticNrv_R': 0.7468304730649833,
        'Parotid_L': 0.8717447502956519,
        'Parotid_R': 0.8573483875851204,
        'Pituitary': 0.6919606441338219,
        'SpinalCord': 0.8117992400145477}
        
        lbls_dict = json.load(open(join(Path(self.plans_file).parent, 'dataset.json')))['labels']
        
        self.organ_weights_list = []
        for _, j in lbls_dict.items():
            if j == 'background':
                self.organ_weights_list.append(1)
            else:
                self.organ_weights_list.append(1/self.organ_weights_dict[j])
            
        self.organ_weights_list = np.array(self.organ_weights_list)**2
        self.organ_weights_list /= self.organ_weights_list.sum()
        
        self.organ_weights_list = torch.Tensor(self.organ_weights_list).to(0)
        
    def initialize_network(self):
        super().initialize_network()
        self.network.ce_loss.weight = self.organ_weights_list
        self.print_to_log_file('WARNING: Using loss weights:', self.organ_weights_list)