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
from model import Model

from loading import create_data_loaders, COLOR_MODE

IPEX = False
KEEP_AWAKE = True
ENABLE_TENSORBOARD = True

DO_TRAINING = True
NUM_EPOCHS = 1

MINIFIED_RUN = False  # only use a small amount of data (for debugging)

start_time = int(time.time())
config = {
    "lr": 0.001,
    "weight_decay": 1e-6,
    "n_hidden_layers": 7,
    "n_hidden_units": 20,
    "kernel_size": 3,
    "help_layer_ids": [2, 3, 5]
}


def config_to_string(config: dict) -> str:
    return f"E{NUM_EPOCHS}__" + "__".join(
        f"{k.replace('_', '')}{('_'.join(f'{x}' for x in v)) if isinstance(v, list) else v}" for k, v in
        config.items()) + f"__{start_time}"

if IPEX:
    import intel_extension_for_pytorch as ipex

if ENABLE_TENSORBOARD and __name__ != "__main__":
    ENABLE_TENSORBOARD=False

if ENABLE_TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=os.path.join("tensorboard", "run_" + config_to_string(config))+"__"+COLOR_MODE)
if torch.cuda.is_available():
    device_id = "cuda:0"
elif torch.is_vulkan_available():
    # Vulkan requires building PyTorch from source with custom preferences (prototype feature) and may not work well.
    # See https://pytorch.org/tutorials/prototype/vulkan_workflow.html?highlight=vulkan and https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-vulkan
    device_id = "vulkan"
else:
    device_id = "cpu"
device = torch.device(device_id)

LOG = logging.getLogger(__name__)


def train(dataloader: DataLoader, validation_loader: DataLoader, config=config) -> Module:
    model = Model(n_inputs=3, n_outputs=3, n_hidden_layers=config["n_hidden_layers"],
                  n_hidden_units=config["n_hidden_units"], kernel_size=config["kernel_size"],
                  help_layer_ids=config["help_layer_ids"])
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
            inner_iterator = tqdm(dataloader, desc=f"Run epoch {epoch_id}", total=len(dataloader), position=1,
                                  leave=False)
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
        for i, data in tqdm(enumerate(test_loader), desc=infotext, total=len(test_loader), position=position,
                            leave=leave):
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
