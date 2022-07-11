from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
import torch
from loading import create_training_test_sets
from model import Model
from training import KEEP_AWAKE, DO_TRAINING, train, test, load, create_example_image
import os

N_MODELS = 10


class EnsembleModule(Module):
    def __init__(self, models: list[Module]):
        super(EnsembleModule, self).__init__()
        self.models = models
        for i,model in enumerate(models):
            for name,param in model.named_parameters():
                self.register_parameter(f"model_{i}_param_{name}",param)

    def forward(self, input: Tensor, known: Tensor):
        # result = torch.zeros((len(self.models),) + input.shape)
        results = []
        for i, model in enumerate(self.models):
            model_result = model(input, known)
            # result[i]=model_result
            results.append(model_result.reshape((1,) + model_result.shape))
        result = torch.cat(results)
        median = torch.median(result, dim=0).values
        stdev = torch.std(result, dim=0)
        lower_bound = median - 2 * stdev
        upper_bound = median + 2 * stdev
        # lower_bound=torch.quantile(input=result,q=float(0.125),dim=0)
        # upper_bound=torch.quantile(input=result,q=float(0.875),dim=0)

        out_of_range = (result < lower_bound) + (result > upper_bound)
        nan_where_out_of_range = 0 / (1 - out_of_range.to(torch.float))
        return torch.nanmean(result + nan_where_out_of_range, dim=0)
        # return torch.median(result, dim=0).values
        # return torch.mean(result, dim=0)
        # TODO find best method


ENSEMBLE_DIR = "ensemble"


def train_ensemble(training_set: Dataset):
    if os.path.exists(ENSEMBLE_DIR):
        for file in os.listdir(ENSEMBLE_DIR):
            os.remove(os.path.join(ENSEMBLE_DIR, file))
    else:
        os.mkdir(ENSEMBLE_DIR)
    for i in tqdm(range(N_MODELS), desc="Training ensemble", position=0):
        training_loader = DataLoader(training_set,
                                     shuffle=True,
                                     batch_size=16,
                                     num_workers=4
                                     )
        model = train(training_loader, None, tqdm_position=1)
        torch.save(model, os.path.join(ENSEMBLE_DIR, f"model_{i}.dmp"))


def load_ensemble():
    models: list[Model] = []
    for i in range(N_MODELS):
        models.append(torch.load(os.path.join(ENSEMBLE_DIR, f"model_{i}.dmp")))
    return EnsembleModule(models)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.backends.cudnn.benchmark = True
    with logging_redirect_tqdm():
        trainingset, testset = create_training_test_sets()
        test_loader = DataLoader(testset,
                                 shuffle=False,
                                 batch_size=16,
                                 num_workers=4
                                 )
        if DO_TRAINING:
            if KEEP_AWAKE:
                from wakepy import set_keepawake, unset_keepawake

                try:
                    set_keepawake(keep_screen_awake=True)
                    train_ensemble(trainingset)
                finally:
                    unset_keepawake()
            else:
                train_ensemble(trainingset)

        model = load_ensemble()

        print(test(model, test_loader))
        create_example_image(model, test_loader)
