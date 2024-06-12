import torch
import torch.nn as nn


def parse_device(device):
    return torch.device(f"cuda:{device[-1]}" if ("gpu" in device.lower()) and (torch.cuda.is_available()) else "cpu")


def compute_conv_layer_output_dim(input_length: int, conv_module: nn.Conv1d) -> int:
    """Compute the output dimension of a convolutional layer.

    Args:
        input_length (int): Length of input.
        conv_module (nn.Conv1d): Convolutional layer.

    Returns:
        int: Output dimension of the convolutional layer.
    """
    return (
        (
            input_length
            + 2 * conv_module.padding[0]  # type: ignore
            - conv_module.dilation[0] * (conv_module.kernel_size[0] - 1)
            - 1
        )
        // conv_module.stride[0]
        + 1
    ) * conv_module.out_channels


def compute_num_conv_features(model: nn.Module) -> int:
    """Compute the number of features output by the convolutional layers.

    Args:
        model (nn.Module): The model containing the convolutional layers.

    Returns:
        int: Number of features output by the convolutional layers.
    """
    temp = 0
    for conv_module in model.conv_list:
        temp += compute_conv_layer_output_dim(model.input_length, conv_module)
    return temp * model.N_receptors


def init_conv_layers(
    input_length: int,
    input_channels: int,
    conv_output_channels: int,
    conv_layer_fraction_widths: list[float],
    conv_stride: int,
    conv_bias: bool = True,
) -> nn.ModuleList:
    """Creates a list of 1D convolutional layers with the specified parameters.
    The multiple filters are created with different kernel sizes, as specified by the
    list of `conv_layer_widths` (which specifies the fraction of the input length that
    each filter kernel should have).

    Args:
        input_length (int): Length of the input data.
        input_channels (int): Number of channels in the input data.
        conv_output_channels (int): Number of output channels for each convolutional layer.
        conv_layer_widths (list[float]): List of fractions of the input length for each filter kernel.
        conv_stride (int): Stride for the convolutional layers.
        conv_bias (bool, optional): Whether the conv layers have a bias term. Defaults to True.

    Returns:
        nn.ModuleList: Torch module list containing the convolutional layers.
    """
    conv_layers = nn.ModuleList([])
    for conv_layer_fraction_width in conv_layer_fraction_widths:
        conv_layers.append(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=conv_output_channels,
                kernel_size=int(input_length * conv_layer_fraction_width),
                stride=conv_stride,
                bias=conv_bias,
            )
        )
    return conv_layers
