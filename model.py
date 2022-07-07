import torch
from torch import Tensor
from torch.nn import Module, Conv2d
from torch.nn.functional import relu_


class Model(Module):
    def __init__(self, n_inputs: int, n_outputs: int, n_hidden_layers: int, n_hidden_units: int, kernel_size: int,
                 help_layer_ids=[]):
        super(Model, self).__init__()
        n_inputs = n_inputs + 3
        self.help_layer_ids = help_layer_ids
        # self.input_layer = Conv2d(in_channels=n_inputs, out_channels=n_hidden_units, kernel_size=kernel_size,
        #                          padding=int(kernel_size / 2))
        # self.known_layer = Conv2d(in_channels=n_inputs, out_channels=n_hidden_units, kernel_size=kernel_size,
        #                          padding=int(kernel_size / 2))
        hidden_layers = []
        for i in range(n_hidden_layers):

            layer = Conv2d(in_channels=n_inputs,
                           out_channels=n_hidden_units,
                           kernel_size=kernel_size,
                           padding=int(kernel_size / 2))
            for j, param in enumerate(layer.parameters()):
                self.register_parameter(f"hidden_{i}_{j}", param)
            hidden_layers.append(layer)
            n_inputs = n_hidden_units
            if i in help_layer_ids:
                n_inputs = n_inputs + 6

        hidden_layers.append(
            Conv2d(in_channels=n_inputs, out_channels=n_outputs, kernel_size=kernel_size,
                   padding=int(kernel_size / 2)))
        self.hidden_layers = hidden_layers

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input: Tensor, known: Tensor):
        # alternative to known (for concatenation) - only contains one dimension
        # known_reduced = known[:, 0, :, :].reshape((known.shape[0], 1, known.shape[2], known.shape[3]))

        known_inverted = -1 * (known - 1)

        x = torch.cat((known, input), dim=1)
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            relu_(x)
            if i < len(self.hidden_layers) - 1 and i in self.help_layer_ids:
                x = torch.cat((x, known, input), dim=1)
        x = x * known_inverted + input

        return x
