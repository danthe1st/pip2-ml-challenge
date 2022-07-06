import torch

from loading import create_data_loaders
from model import Model, load as load_model
import numpy as np
import numpy.typing as npt
import dill as pkl

from torch.nn import Module
from torch.utils.data import DataLoader

from tqdm import tqdm


def convert_model_output(model_output: npt.NDArray, known: npt.NDArray):
    arr = np.array(model_output * 255)
    known_indices = np.array(np.where(known[0] == 0))
    ret = np.empty((model_output.shape[0], len(known_indices[0])), dtype=np.uint8)

    for i, index_index in enumerate(range(known_indices.shape[1])):
        ret[:, i] = arr[:, known_indices[0, index_index], known_indices[1, index_index]]
    return ret.flatten()


def save(data, file):
    with open(file, "wb") as fh:
        pkl.dump(data, fh)

def run_model(model: Module,
              dataloader: DataLoader[tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float], str]]):
    actual_result = []
    expected = []
    for input, known, target, file in tqdm(dataloader, desc="running model", total=len(dataloader)):
        output = model(input.to(torch.float32), known.to(torch.float32))
        for i in range(output.shape[0]):
            if len(target) != 0:
                expected.append(
                    convert_model_output(target[i, :, :, :].detach().numpy(), known[i, :, :, :].detach().numpy()))
            actual_result.append(
                convert_model_output(output[i, :, :, :].detach().numpy(), known[i, :, :, :].detach().numpy()))
    return actual_result, expected


if __name__ == "__main__":
    training_loader, test_loader = create_data_loaders()
    model = load_model()
    result, expected = run_model(model, test_loader)
    save(result, "model_output.pkl")
    save(expected, "model_expected.pkl")
