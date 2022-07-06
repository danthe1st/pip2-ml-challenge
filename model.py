import logging
import os.path
import time

import numpy as np
import torch
from PIL import Image, PyAccess
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import MSELoss
from torch.nn import Module
from torch.nn.functional import relu_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from loading import create_data_loaders

IPEX = False
KEEP_AWAKE = True
ENABLE_TENSORBOARD = True

DO_TRAINING = True
NUM_EPOCHS = 10

MINIFIED_RUN=False # only use a small amount of data (for debugging)


config = {
    "lr": 0.001,
    "weight_decay": 1e-6,
    "n_hidden_layers": 7,
    "n_hidden_units": 20,
    "kernel_size": 3,
    "help_layer_ids":[2,3]
}
def config_to_string(config: dict) -> str:
    return "__".join(f"{k.replace('_','')}{('_'.join(f'{x}' for x in v)) if isinstance(v,list) else v}" for k, v in config.items())
#tuning_config = {
#    "lr": tune.grid_search([0.05, 0.01, 0.001, 0.0001]),
#    "weight_decay": tune.grid_search([1e-6, 1e-5, 1e-4]),
#    "n_hidden_layers": 5,
#    #"n_hidden_layers": tune.grid_search([5, 6, 7, 8]),
#    "n_hidden_units": 15,
#    #"n_hidden_units": tune.grid_search([10, 15, 20]),
#    "kernel_size": 9
#    #"kernel_size": tune.grid_search([7, 9, 11])
#}

if IPEX:
    import intel_extension_for_pytorch as ipex

if ENABLE_TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=os.path.join("tensorboard", "run_" + config_to_string(config)))
if torch.cuda.is_available():
    device_id = "cuda:0"
elif torch.is_vulkan_available():
    # Vulkan requires building PyTorch from source with custom preferences (prototype feature) and may not work well.
    # See https://pytorch.org/tutorials/prototype/vulkan_workflow.html?highlight=vulkan and https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-vulkan
    device_id = "vulkan"
else:
    device_id = "cpu"
device = torch.device(device_id)


class Model(Module):

    def __init__(self, n_inputs: int, n_outputs: int, n_hidden_layers: int, n_hidden_units: int, kernel_size: int, help_layer_ids=[]):
        super(Model, self).__init__()
        n_inputs=n_inputs+3
        self.help_layer_ids=help_layer_ids
        # self.input_layer = Conv2d(in_channels=n_inputs, out_channels=n_hidden_units, kernel_size=kernel_size,
        #                          padding=int(kernel_size / 2))
        # self.known_layer = Conv2d(in_channels=n_inputs, out_channels=n_hidden_units, kernel_size=kernel_size,
        #                          padding=int(kernel_size / 2))
        hidden_layers = []
        for i in range(n_hidden_layers):

            layer = Conv2d(in_channels=n_inputs+6 if i-1 in help_layer_ids else n_inputs,
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

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input: Tensor, known: Tensor):
        known_reduced=known[:,0,:,:].reshape((known.shape[0],1,known.shape[2],known.shape[3]))
        known_inverted = -1 * (known - 1)

        x = torch.cat((known,input),dim=1)
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            relu_(x)
            if i < len(self.hidden_layers) - 1 and i in self.help_layer_ids:
                x = torch.cat((x, known, input), dim=1)
        x = x * known_inverted + input

        return x


LOG = logging.getLogger(__name__)


def train(dataloader: DataLoader, validation_loader: DataLoader, config=config) -> Module:
    model = Model(n_inputs=3, n_outputs=3, n_hidden_layers=config["n_hidden_layers"],
                  n_hidden_units=config["n_hidden_units"], kernel_size=config["kernel_size"], help_layer_ids=config["help_layer_ids"])
    model.to(device)
    model.train()
    # optimizer = SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if IPEX:
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
    loss_fn = MSELoss()
    for epoch_id in tqdm(range(NUM_EPOCHS), "Training model", NUM_EPOCHS, position=0):
        model.train()
        inner_iterator = dataloader
        if not MINIFIED_RUN:
            inner_iterator = tqdm(dataloader, desc=f"Run epoch {epoch_id}", total=len(dataloader), position=1, leave=False)
        for iteration, data in enumerate(inner_iterator):
            if MINIFIED_RUN and iteration > 200:
                break
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
            optimizer.zero_grad(set_to_none=True)
            # LOG.info(f"loss: {loss.item()}")
            if ENABLE_TENSORBOARD and iteration % 50 == 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(),
                                  global_step=epoch_id * len(dataloader) + iteration)
                for j, (name, param) in enumerate(model.named_parameters()):
                    writer.add_histogram(tag=f"training/param{j} ({name})", values=param.cpu(),
                                         global_step=epoch_id * len(dataloader) + iteration)
        if ENABLE_TENSORBOARD:
            validation_loss = test(model, validation_loader, infotext=f"Validating model after epoch {epoch_id}",
                                   position=1, leave=False)
            writer.add_scalar(tag="validation/loss", scalar_value=validation_loss,
                              global_step=(epoch_id + 1) * len(dataloader) - 1)
    return model


def test(model: Module, test_loader: DataLoader, infotext="Testing model", position=0, leave=True):
    with torch.no_grad():
        model.eval()
        loss_fn = MSELoss()
        loss_fn.requires_grad_(False)
        losses = np.zeros((51 if MINIFIED_RUN else len(test_loader),))
        for i, data in tqdm(enumerate(test_loader), desc=infotext, total=len(test_loader), position=position,leave=leave):
            if MINIFIED_RUN and i >= 50:
                break
            input, known, target, file = data
            input = input.to(torch.float)
            known = known.to(torch.float)
            output = model(input, known)
            loss = loss_fn(output, target)

            losses[i] = loss
        # print(losses)
        return np.mean(losses)


def pixels_to_image(pixels: torch.Tensor, filename):
    im = Image.new("RGB", (100, 100))
    pix: PyAccess = im.load()
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

def main():
    logging.basicConfig(level=logging.INFO)
    torch.backends.cudnn.benchmark = True
    with logging_redirect_tqdm():
        training_loader, test_loader = create_data_loaders()
        if DO_TRAINING:
            if KEEP_AWAKE:
                from wakepy import set_keepawake, unset_keepawake

                try:
                    set_keepawake(keep_screen_awake=True)
                    model = train(training_loader, test_loader)
                finally:
                    set_keepawake(keep_screen_awake=False)
            else:
                model = train(training_loader, test_loader)
            torch.save(model, "model.dmp")

            if KEEP_AWAKE:
                unset_keepawake()
        else:
            model = load()
        print(test(model, test_loader))
        create_example_image(model, test_loader)

    if ENABLE_TENSORBOARD:
        writer.close()


if __name__ == "__main__":
    main()