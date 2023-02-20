python3 /media/medical/gasperp/projects/nnUnet_clone/nnUNet/nnunet/results/register_patch_and_predict.py \
    -t "208" \
    -conf "3d_fullres" \
    -f "all" \
    -tr "nnUNetTrainerV2_DPnoMirroringAxis2redRot_dualPath_sep2" \
    --save_seg_masks \
    --gpus 0 1 \
    --phases_to_predict "test" \
    --organs_of_interest "OpticChiasm" "Glnd_Lacrimal_L" "Glnd_Lacrimal_R" "Parotid_L" "Parotid_R" "Pituitary" "Cochlea_L" "Cochlea_R" "OpticNrv_L" "OpticNrv_R" "Glnd_Submand_L" "Glnd_Submand_R"