python3 /media/medical/gasperp/projects/nnUnet_clone/nnUNet/nnunet/results/predict.py \
    -t  107 \
    -conf "3d_fullres" \
    -f 0 \
    -tr "nnUNetTrainerV2_DP_change_normalization_learn_window" \
    --save_seg_masks \
    --gpus 0 1 \
    --phases_to_predict "train"