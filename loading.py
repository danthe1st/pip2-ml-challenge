import os.path

from typing import Tuple

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch
from glob import glob
from PIL import Image
import colorsys
from ex4 import ex4 as convert_image
import dill as pkl

# set seeds
np.random.seed(0)
torch.random.manual_seed(0)


# offset: uniform random between 0 and 8 (both inclusive)
# spacings: uniform random between 2 and 6 (both inclusive)
# max target_array len: 29232

# testset: images have shape (3,100,100)

class ChallengeDataset(Dataset[Tuple[npt.NDArray, npt.NDArray, npt.NDArray, str]]):
    def __init__(self):
        with open("testset.pkl", "rb") as fh:
            data = pkl.load(fh)
            self.input_arrays = data["input_arrays"]
            self.known_arrays = data["known_arrays"]
            self.offsets = data["offsets"]
            self.spacings = data["spacings"]
            self.sample_ids = data["sample_ids"]

    def __len__(self):
        return len(self.input_arrays)

    def __getitem__(self, item):
        return self.input_arrays[item] / 255, self.known_arrays[item], [], f"{self.sample_ids[item]}.jpg"


class TrainingDataset(Dataset[Tuple[npt.NDArray, npt.NDArray, npt.NDArray, str]]):
    def __init__(self):
        self.files = glob(os.path.join("training", "*", "*.jpg"), recursive=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        num_offsets = 9
        num_spacings = 5
        offset = (item % num_offsets, (item / num_offsets) % num_offsets)
        spacing = ((item / (num_offsets * num_offsets)) % num_spacings + 2,
                   (item / (num_offsets * num_offsets * num_spacings)) % num_spacings + 2)
        file = self.files[item]
        with open(file, "rb") as fh:
            img = Image.open(fh)
            data = np.array(img.resize((100, 100), Image.BILINEAR))
        input_array, known_array, target_array = convert_image(data, offset, spacing)
        target_image = np.transpose(data, (2, 0, 1))
        return input_array / 255, known_array, target_image / 255, os.path.relpath(file, "training")
    # input_array, known_array, target_array


def load_indices(dataset: Dataset) -> Tuple[npt.NDArray[int], npt.NDArray[int]]:
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)
    split_point = int(len(all_indices) * 4 / 5)
    return all_indices[0:split_point], all_indices[split_point:len(all_indices)]


def create_training_test_sets() -> Tuple[
    Dataset[Tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float], str]], Dataset[
        Tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float], str]]]:
    dataset = TrainingDataset()
    training_indices, test_indices = load_indices(dataset)
    trainingset = Subset(dataset, indices=training_indices)
    testset = Subset(dataset, indices=test_indices)
    return trainingset, testset


def create_data_loaders():
    trainingset, testset = create_training_test_sets()
    training_loader = DataLoader(trainingset,
                                 shuffle=False,
                                 batch_size=16,
                                 num_workers=8
                                 )
    test_loader = DataLoader(testset,
                             shuffle=False,
                             batch_size=16,
                             num_workers=8
                             )
    return training_loader, test_loader

def create_challenge_data_loader():
    return DataLoader(ChallengeDataset(),
                      shuffle=False,
                      batch_size=16,
                      num_workers=8)

if __name__ == "__main__":
    training_loader, test_loader = create_data_loaders()
    for x in training_loader:
        # 1st dimension-->minibatch
        # 2nd dimension-->color channel
        # 3rd dimension-->x
        # 4th dimension-->y
        input, known, target, file = x
        print(input.shape, known.shape, target.shape)
