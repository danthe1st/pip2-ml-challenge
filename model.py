import logging
import os.path

import numpy as np
import numpy.typing as npt
from PIL import Image
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import MSELoss
from torch.nn import Module
from torch.nn.functional import relu_
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from loading import create_data_loaders

IPEX = False
KEEP_AWAKE = False

DO_TRAINING = False
NUM_EPOCHS = 40


class Model(Module):
    def __init__(self, n_inputs: int, n_outputs: int, n_hidden_layers: int, n_hidden_units: int, kernel_size: int):
        super(Model, self).__init__()

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
            # hidden_layers.append(ReLU())
            n_inputs = n_hidden_units

        hidden_layers.append(
            Conv2d(in_channels=n_hidden_units, out_channels=n_outputs, kernel_size=kernel_size,
                   padding=int(kernel_size / 2)))
        # self.hidden_layers = Sequential(*hidden_layers)
        self.hidden_layers = hidden_layers

        known_weights_tensor = torch.ones(size=(len(hidden_layers),), dtype=torch.float32)
        self.known_weights = torch.nn.Parameter(data=known_weights_tensor, requires_grad=True)

        custom_weights_on_known_pixels_tensor = torch.ones(size=(len(hidden_layers),), dtype=torch.float32)
        self.custom_weights_on_known_pixels = torch.nn.Parameter(data=custom_weights_on_known_pixels_tensor,
                                                                 requires_grad=True)

        for param in self.parameters():
            param.requires_grad = True

        self.manual_help = n_hidden_units == 3

    def forward(self, input: Tensor, known: Tensor):

        # x = self.input_layer(input)
        # relu_(x)
        # k = self.known_layer(known)
        # relu_(k)
        # hidden_in = (x + k) / 2
        # known_inverted = (-(known[:,0,:,:] - 1))
        known_inverted = (-(known - 1))
        # known_inverted = known_inverted * known_inverted  # abs
        # known_inverted=known_inverted.reshape(known_inverted.shape[0],1,known_inverted.shape[1],known_inverted.shape[2])
        ## return self.hidden_layers(hidden_in)
        # result=[]
        # for channel in range(input.shape[1]):
        #    reshaped_input=input[:,channel,:,:].reshape(input.shape[0],1,input.shape[2],input.shape[3])
        #    x = reshaped_input
        #    for i, layer in enumerate(self.hidden_layers):
        #        x = layer(x)
        #        relu_(x)
        #        x = x * known_inverted + reshaped_input * self.known_weights[i]
        #    x = x * known_inverted + reshaped_input
        #    result.append(x)
        # x=torch.cat(result,dim=1)

        x = input
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if (self.manual_help):
                x = x * known_inverted * (1 - self.custom_weights_on_known_pixels[i]) + input * self.known_weights[i]
            relu_(x)
        x = x * known_inverted + input

        return x


LOG = logging.getLogger(__name__)


def train(dataloader: DataLoader) -> Module:
    model = Model(n_inputs=3, n_outputs=3, n_hidden_layers=7, n_hidden_units=20, kernel_size=3)
    # optimizer = SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    if IPEX:
        import intel_extension_for_pytorch as ipex
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
    loss_fn = MSELoss()
    for epoch_id in tqdm(range(NUM_EPOCHS), "Training model", NUM_EPOCHS, position=0):
        for iteration, data in enumerate(
                tqdm(dataloader, desc=f"Run epoch {epoch_id}", total=len(dataloader), position=1, leave=False)):
            # 1st dimension-->minibatch
            # 2nd dimension-->color channel
            # 3rd dimension-->x
            # 4th dimension-->y

            # file-->array of file names (for debugging)
            input, known, target, file = data
            input = input.to(torch.float)
            input.requires_grad_()
            known = known.to(torch.float)
            known.requires_grad_()
            target = target.to(torch.float)
            result = model(input, known)
            loss = loss_fn(result, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # LOG.info(f"loss: {loss.item()}")
    return model


def test(model: Module, test_loader: DataLoader, position=0, leave=True):
    loss_fn = MSELoss()
    loss_fn.requires_grad_(False)
    losses = np.zeros((len(test_loader),))
    for i, data in tqdm(enumerate(test_loader), desc="Testing model", total=len(test_loader), position=position,
                        leave=leave):
        input, known, target, file = data
        input = input.to(torch.float)
        known = known.to(torch.float)
        output = model(input, known)
        loss = loss_fn(output, target)
        losses[i] = loss
    # print(losses)
    return np.mean(losses)


def pixels_to_image(pixels: npt.NDArray, filename):
    im = Image.new("RGB", (100, 100))
    pix = im.load()
    for x in range(100):
        for y in range(100):
            pix[x, y] = tuple(pixels[:, x, y].detach().numpy())
    im.save(filename, "JPEG")


def load() -> Module:
    return torch.load("model.dmp")


def create_example_image(model: Module, dataloader: DataLoader):
    data = next(iter(dataloader))
    input, known, target, file = data
    output = model(input.to(torch.float), known.to(torch.float))
    pixels_to_image((input[0, :, :, :] * 255).to(torch.int), "in.jpg")
    pixels_to_image((output[0, :, :, :] * 255).to(torch.int), "out.jpg")
    if target is not None:
        pixels_to_image((target[0, :, :, :] * 255).to(torch.int), "target.jpg")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        training_loader, test_loader = create_data_loaders()
        if DO_TRAINING:
            if KEEP_AWAKE:
                from wakepy import set_keepawake, unset_keepawake

                try:
                    model = train(training_loader)
                finally:
                    set_keepawake(keep_screen_awake=False)
            else:
                model = train(training_loader)
            torch.save(model, "model.dmp")

            if KEEP_AWAKE:
                unset_keepawake()
        else:
            model = load()
        print(test(model, test_loader))
        create_example_image(model, test_loader)
