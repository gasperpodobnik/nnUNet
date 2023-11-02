from dynamic_network_architectures.architectures.unet import (
    PlainConvUNet,
    ResidualEncoderUNet,
)
from dynamic_network_architectures.architectures.unet_multi_encoder import (
    PlainConvUNetMultiEncoder,
)
from dynamic_network_architectures.architectures.unet_Dou import (
    PlainConvUNetSeparateNorm,
    PlainConvUNetSeparateNormV2,
)
from dynamic_network_architectures.architectures.unet_STN import PlainConvUNetSTN
from dynamic_network_architectures.architectures.unet_Dou_plus_STN import (
    PlainConvUNetDouPlusSTN,
)
from dynamic_network_architectures.architectures.unet_modal import (
    PlainConvUNet_modal0,
    PlainConvUNet_modal1,
)
from dynamic_network_architectures.architectures.unet2 import (
    PlainConvUNet_singleEncoder2Modals,
)
from dynamic_network_architectures.architectures.unet_CMX import (
    PlainConvUNetSeparateNormCMX,
    PlainConvUNetSeparateEncoderCMX,
    PlainConvUNetSeparateEncoderCMXv2,
    PlainConvUNetSeparateEncoderCMXv3,
    PlainConvUNetSeparateEncoderCMXv3_3modals,
    PlainConvUNetCrossAttnDecoder,
    PlainConvUNetSeparateEncoderCMX_changedSoftmaxAxis,
    PlainConvUNetSeparateEncoderPyTorchNativeAttention,
    PlainConvUNetCrossAttnSoftTissue,
)
from dynamic_network_architectures.building_blocks.helper import (
    get_matching_instancenorm,
    convert_dim_to_conv_op,
)
from dynamic_network_architectures.initialization.weight_init import (
    init_last_bn_before_add_to_0,
)
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from torch import nn


def get_network_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True,
):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = configuration_manager.UNet_class_name
    mapping = {
        "PlainConvUNet": PlainConvUNet,
        "ResidualEncoderUNet": ResidualEncoderUNet,
        "PlainConvUNetMultiEncoder": PlainConvUNetMultiEncoder,
        "PlainConvUNetSeparateNorm": PlainConvUNetSeparateNorm,
        "PlainConvUNetSeparateNormV2": PlainConvUNetSeparateNormV2,
        "PlainConvUNetSTN": PlainConvUNetSTN,
        "PlainConvUNetDouPlusSTN": PlainConvUNetDouPlusSTN,
        "PlainConvUNet_modal0": PlainConvUNet_modal0,
        "PlainConvUNet_modal1": PlainConvUNet_modal1,
        "PlainConvUNetSeparateNormCMX": PlainConvUNetSeparateNormCMX,
        "PlainConvUNetSeparateEncoderCMX": PlainConvUNetSeparateEncoderCMX,
        "PlainConvUNetSeparateEncoderCMXfull": PlainConvUNetSeparateEncoderCMX,
        "PlainConvUNetSeparateEncoderCMXv2": PlainConvUNetSeparateEncoderCMXv2,
        "PlainConvUNetSeparateEncoderCMXv3": PlainConvUNetSeparateEncoderCMXv3,
        "PlainConvUNetSeparateEncoderCMXv3_3modals": PlainConvUNetSeparateEncoderCMXv3_3modals,
        "PlainConvUNetCrossAttnDecoder": PlainConvUNetCrossAttnDecoder,
        "PlainConvUNetSeparateEncoderCMX_changedSoftmaxAxis": PlainConvUNetSeparateEncoderCMX_changedSoftmaxAxis,
        "PlainConvUNetSeparateEncoderPyTorchNativeAttention": PlainConvUNetSeparateEncoderPyTorchNativeAttention,
        "PlainConvUNetCrossAttnSoftTissue": PlainConvUNetCrossAttnSoftTissue,
        "PlainConvUNet_singleEncoder2Modals": PlainConvUNet_singleEncoder2Modals,
    }

    default_kwargs = {
        "conv_bias": True,
        "norm_op": get_matching_instancenorm(conv_op),
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": nn.LeakyReLU,
        "nonlin_kwargs": {"inplace": True},
    }

    kwargs = {
        "PlainConvUNet": default_kwargs,
        "ResidualEncoderUNet": default_kwargs,
        "PlainConvUNetSeparateEncoderCMX": {
            **default_kwargs,
            "reduction": 4,
        },
        "PlainConvUNetSeparateEncoderCMXfull": {
            **default_kwargs,
            "reduction": 1,
        },
        "PlainConvUNetSeparateEncoderCMXv2": {
            **default_kwargs,
            "reduction": 1,
        },
    }
    assert segmentation_network_class_name in mapping.keys(), (
        "The network architecture specified by the plans file "
        "is non-standard (maybe your own?). Yo'll have to dive "
        "into either this "
        "function (get_network_from_plans) or "
        "the init of your nnUNetModule to accomodate that."
    )
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        "n_conv_per_stage"
        if network_class != ResidualEncoderUNet
        else "n_blocks_per_stage": configuration_manager.n_conv_per_stage_encoder,
        "n_conv_per_stage_decoder": configuration_manager.n_conv_per_stage_decoder,
    }
    # network class name!!
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[
            min(
                configuration_manager.UNet_base_num_features * 2**i,
                configuration_manager.unet_max_num_features,
            )
            for i in range(num_stages)
        ],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs.get(segmentation_network_class_name, default_kwargs),
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_in_M = round(trainable_params / 1e6, 2)
    total_params_in_M = round(total_params / 1e6, 2)
    params_string = f"Number of trainable/total parameters in M: {trainable_params_in_M}/{total_params_in_M}"
    print(params_string)
    return model
