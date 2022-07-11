import os.path
from glob import glob
from typing import Tuple, Literal

import dill as pkl
import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms
from PIL import Image, PyAccess
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset

from ex4 import ex4 as convert_image


# set seeds
def reset_seeds():
    np.random.seed(0)
    torch.random.manual_seed(0)


reset_seeds()

COLOR_MODE = "RGB"


# offset: uniform random between 0 and 8 (both inclusive)
# spacings: uniform random between 2 and 6 (both inclusive)
# max target_array len: 29232

# testset: images have shape (3,100,100)

class ChallengeDataset(Dataset[tuple[npt.NDArray, npt.NDArray, npt.ArrayLike, str]]):
    def __init__(self):
        with open("testset.pkl", "rb") as fh:
            data = pkl.load(fh)
        self.input_arrays: tuple[npt.NDArray] = data["input_arrays"]
        self.known_arrays: tuple[npt.NDArray] = data["known_arrays"]
        self.offsets: npt.NDArray = data["offsets"]
        self.spacings: npt.NDArray = data["spacings"]
        self.sample_ids: npt.NDArray = data["sample_ids"]

    def __len__(self):
        return len(self.input_arrays)

    def __getitem__(self, item: int) -> tuple[npt.NDArray, npt.NDArray, npt.ArrayLike, str]:
        input_array=self.input_arrays[item]
        if COLOR_MODE != "RGB":
            input_array=np.transpose(change_image_format(np.transpose(input_array, (1, 2, 0)), "RGB",COLOR_MODE), (2, 0, 1))
        input_array=np.array(input_array,dtype=float)
        return input_array / 255, self.known_arrays[item], [], f"{self.sample_ids[item]}.jpg"


class TrainingDataset(Dataset[tuple[npt.NDArray, npt.NDArray, npt.ArrayLike, str]]):
    def __init__(self):
        self.files: list[str] = glob(os.path.join("training", "*", "*.jpg"), recursive=True)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, str]:
        num_offsets = 9
        num_spacings = 5
        offset = (np.random.randint(0, num_offsets), np.random.randint(0, num_offsets))
        spacing = (np.random.randint(2, num_spacings + 2), np.random.randint(2, num_spacings + 2))
        file = self.files[item]
        with open(file, "rb") as fh:
            # img = self.transforms(Image.open(fh))
            img = Image.open(fh)
            # img=img.convert("HSV")
            data = np.array(img.resize((100, 100), Image.BILINEAR))
            if COLOR_MODE != "RGB":
                data = change_image_format(data, "RGB", COLOR_MODE)
        input_array, known_array, target_array = convert_image(data, offset, spacing)
        target_image = np.transpose(data, (2, 0, 1))
        return input_array / 255, known_array, target_image / 255, os.path.relpath(file, "training")
    # input_array, known_array, target_array


def change_image_format(pixels: npt.NDArray,
                        in_format: Literal[
                            "1", "CMYK", "F", "HSV", "I", "L", "LAB", "P", "RGB", "RGBA", "RGBX", "YCbCr"],
                        out_format: Literal[
                            "1", "CMYK", "F", "HSV", "I", "L", "LAB", "P", "RGB", "RGBA", "RGBX", "YCbCr"]) -> npt.NDArray:
    img = Image.new(in_format, (100, 100))
    pix: PyAccess = img.load()
    for x in range(100):
        for y in range(100):
            pix[x, y] = tuple(pixels[x, y, :])
    img = img.convert(out_format)
    return np.array(img)


def load_indices(dataset: Dataset[tuple[npt.NDArray, npt.NDArray, npt.ArrayLike, str]]) -> Tuple[npt.NDArray[int], npt.NDArray[int]]:
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


def create_data_loaders() -> tuple[DataLoader,DataLoader]:
    trainingset, testset = create_training_test_sets()
    training_loader = DataLoader(trainingset,
                                 shuffle=False,
                                 batch_size=16,
                                 num_workers=4
                                 )
    test_loader = DataLoader(testset,
                             shuffle=False,
                             batch_size=16,
                             num_workers=4
                             )
    return training_loader, test_loader


def create_challenge_data_loader() -> DataLoader:
    return DataLoader(ChallengeDataset(),
                      shuffle=False,
                      batch_size=16,
                      num_workers=4)


if __name__ == "__main__":
    training_loader, test_loader = create_data_loaders()
    for x in training_loader:
        # 1st dimension-->minibatch
        # 2nd dimension-->color channel
        # 3rd dimension-->x
        # 4th dimension-->y
        input, known, target, file = x
        print(input.shape, known.shape, target.shape)
