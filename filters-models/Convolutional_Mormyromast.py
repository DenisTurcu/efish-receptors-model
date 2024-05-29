import torch
import torch.nn as nn
import sys

sys.path.append("../filters-models")

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
        conv_bias: bool = True,
        adapt_alpha: float = 0.99,
    ):
        super(ConvMormyromast, self).__init__()

        self.input_length = input_length
        self.N_receptors = N_receptors
        self.conv_list = init_conv_layers(
            input_length, input_channels, conv_output_channels, conv_layer_fraction_widths, conv_stride, conv_bias
        )
        self.num_conv_feats = compute_num_conv_features(self)
        self.bn = nn.BatchNorm1d(num_features=self.num_conv_feats)

        # adaptation filters based on BatchNorm computation  # self.conv_layer_output_dim * N_receptors
        self.adapt_As = torch.ones(self.num_conv_feats)
        self.adapt_Bs = torch.zeros(self.num_conv_feats)
        self.adapt_gammas = nn.Parameter(torch.ones(self.num_conv_feats))
        self.adapt_betas = nn.Parameter(torch.zeros(self.num_conv_feats))
        self.adapt_coeff = (1 - adapt_alpha) / 2

    def forward(self, x: torch.Tensor, mode: str = "regular", use_bn: bool = True) -> torch.Tensor:
        """Forward pass of the ConvMormyromast model.

        Args:
            x (torch.Tensor): Input tensor of shape (N_samples, N_receptors, N_time_points).
            mode (str, optional): Forward mode - `regular` or `adaptive`. Defaults to "regular".
            use_bn (bool, optional): Whether to apply BatchNorm. Defaults to True.

        Raises:
            ValueError: If an invalid mode is provided.

        Returns:
            torch.Tensor: Output tensor of shape (N_samples, N_features).
        """
        if mode.lower() == "regular":
            return self.forward_regular(x, use_bn)
        elif mode.lower() == "adaptive":
            return self.forward_adaptive(x)
        else:
            raise ValueError("Invalid mode. Choose 'regular' or 'adaptive'.")

    def forward_regular(self, x: torch.Tensor, use_bn=True) -> torch.Tensor:
        """Forward pass of the ConvMormyromast model via the regular method (no adaptation).

        Args:
            x (torch.Tensor): Input tensor of shape (N_samples, N_receptors, N_time_points).
            use_bn (bool, optional): Whether to apply BatchNorm. Defaults to True.

        Returns:
            torch.Tensor: Output tensor of shape (N_samples, N_features).
        """
        assert x.shape[2] == self.input_length, "Input must match conv filter length"
        out_feats = torch.cat(
            [conv_module(x.reshape(x.shape[0] * x.shape[1], 1, x.shape[2])) for conv_module in self.conv_list], dim=2
        )
        out_feats = out_feats.reshape(x.shape[0], -1)
        if use_bn:
            return self.bn(out_feats)
        return out_feats

    def forward_adaptive(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ConvMormyromast model via the adaptive method.

        Args:
            x (torch.Tensor): Input tensor of shape (N_samples, N_receptors, N_time_points).

        Returns:
            torch.Tensor: Output tensor of shape (N_samples, N_features).
        """
        out_feats = torch.zeros([x.shape[0], self.num_conv_feats])
        for i in range(x.shape[0]):
            # xx needs to be shape N_receptors x Time_length (i.e. one sample at a time)
            xx = x[i]

            # compute the output of the convolutional layers (before adjusting based on adaptation parameters)
            out_before = torch.cat([conv_module(xx.unsqueeze(1)) for conv_module in self.conv_list], dim=2).reshape(-1)

            # adjust the output based on the adaptation parameters
            out_after = self.adapt_As * (out_before - self.adapt_Bs)

            # store the output for this sample
            out_feats[i] = out_after

            ########################################
            # adaptation update of the receptor cell
            temp = torch.pow((out_after - self.adapt_betas) / self.adapt_gammas, 2)
            self.adapt_Bs = self.adapt_Bs + self.adapt_coeff * (
                2 * (out_before - self.adapt_Bs) - self.adapt_betas / self.adapt_As * (1 + temp)
            )
            self.adapt_As = self.adapt_As * (1 + self.adapt_coeff * (1 - temp))
            ########################################

        return out_feats

    def init_adaptation_values_from_BatchNorm(self):
        """Initialize the adaptation values based on the BatchNorm parameters."""
        mean = self.bn.running_mean.data  # type: ignore
        variance = self.bn.running_var.data  # type: ignore
        gamma = self.bn.weight.data
        beta = self.bn.bias.data
        eps = self.bn.eps

        self.adapt_gammas = nn.Parameter(gamma)
        self.adapt_betas = nn.Parameter(beta)
        self.adapt_As = gamma / torch.sqrt(variance + eps)
        self.adapt_Bs = mean - beta / self.adapt_As
        pass
