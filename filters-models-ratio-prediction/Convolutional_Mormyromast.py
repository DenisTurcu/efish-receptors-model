import torch
import torch.nn as nn

from helpers_convolutional_filters import compute_num_conv_features, init_conv_layers  # noqa: E402


class ConvMormyromast(nn.Module):
    """Convolutional Mormyromast model, with continual adaptation that adjusts the
    convolutional filters' responses based on streaming data.

    Args:
        nn (torch.nn Module): This class inherits from the torch.nn.Module class.
    """

    def __init__(
        self,
        input_length: int = 1000,
        input_channels: int = 1,
        conv_layer_fraction_widths: list[float] = [1, 1],
        conv_output_channels: int = 1,
        conv_stride: int = 25,
        N_receptors: int = 1,
        conv_bias: bool = False,
    ):
        super(ConvMormyromast, self).__init__()

        self.input_length = input_length
        self.N_receptors = N_receptors
        self.conv_list = init_conv_layers(
            input_length, input_channels, conv_output_channels, conv_layer_fraction_widths, conv_stride, conv_bias
        )

    def forward(self, distorted_stim: torch.Tensor, base_stim: torch.Tensor, use_bn=True) -> torch.Tensor:
        """Forward pass of the ConvMormyromast model.

        Args:
            x (torch.Tensor): Input tensor of shape (N_samples, N_receptors, N_time_points).
            use_bn (bool, optional): Whether to apply BatchNorm. Defaults to True.

        Returns:
            torch.Tensor: Output tensor of shape (N_samples, N_features).
        """
        assert distorted_stim.shape[2] == self.input_length, "Input must match conv filter length"
        assert base_stim.shape[2] == self.input_length, "Input must match conv filter length"
        out_distorted = torch.cat(
            [
                conv_module(
                    distorted_stim.reshape(
                        distorted_stim.shape[0] * distorted_stim.shape[1], 1, distorted_stim.shape[2]
                    )
                )
                for conv_module in self.conv_list
            ],
            dim=2,
        )
        out_distorted = out_distorted.reshape(distorted_stim.shape[0], -1)
        out_base = torch.cat(
            [
                conv_module(base_stim.reshape(base_stim.shape[0] * base_stim.shape[1], 1, base_stim.shape[2]))
                for conv_module in self.conv_list
            ],
            dim=2,
        )
        out_base = out_base.reshape(base_stim.shape[0], -1)
        return out_distorted / out_base - 1
