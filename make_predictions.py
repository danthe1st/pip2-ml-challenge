import os

from loading import create_challenge_data_loader
from prepare_validation import save, run_model
from model import Model
from training import load as load_model
import ensemble
import torch
from training import pixels_to_image

SAVE_PREDICTED_IMAGES=False



if __name__ == "__main__":
    dataloader = create_challenge_data_loader()
    #model: Model = load_model()
    model = ensemble.load_ensemble()
    result, _ = run_model(model, dataloader)
    save(result, "challenge_predictions.pkl")
    if SAVE_PREDICTED_IMAGES:
        if not os.path.exists("challenge_predictions"):
            os.mkdir("challenge_predictions")
        for input, known, target, file in dataloader:
            for i in range(input.shape[0]):
                output = model(input.to(torch.float), known.to(torch.float))
                pixels_to_image((input[i, :, :, :] * 255).to(torch.int), f"challenge_predictions/in_{file[i]}")
                pixels_to_image((output[i, :, :, :] * 255).to(torch.int), f"challenge_predictions/out_{file[i]}")
