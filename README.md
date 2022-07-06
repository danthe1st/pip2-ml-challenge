# ML Challenge

The code and model of my submissions for the ML challenge of the `Programming in Python II` course.

The goal is to complete images where only a fraction of the pixels are present

### Files

#### Source files

- `ex4.py` - the solution to exercise 4
- `loading.py` - takes care of any sort of data loading
    - the main in there is for testing
- `model.py` - contains the model architecture and functions for training/testing the model (as well as saving the model
  to an image file)
    - The main in there trains and tests a model
    - The following variables can be used for configuration
        - `IPEX` enables Intel extensions for Pytorch (should only be set to `True` if that module is installed)
        - `KEEP_AWAKE` enables an experimental script that tries to stop the computer from suspending while the model is
          training
        - `DO_TRAINING` if this variable is set to `True`, it will train a new model and save it to a file
          named `model.dmp`.
        - `NUM_EPOCHS` the number of epochs to run the algorithm with
          If it is set to `false`, it will only validate the model.
        - `config` model configuration/hyperparameters
- `prepare_validation.py` loads the validation data, runs the model on that and creates a pickle file in the challenge
  output format as well as a file with target predictions
- `make_predictions.py` loads the challenge dataset from a file called `testset.pkl`, runs the model on that and saves
  the prediction to a file `challenge_predictions.pkl` (in the output format).
    - The following variables can be used for configuration:
        - `SAVE_PREDICTED_IMAGES` if set to `True`, saves predicted images to a directory named `challenge_predictions`
          in order to check whether everything worked

#### other files and directories

- `model.dmp` - the trained model
- `training` - all images used for training should be in that folder
- `challenge_prediction` - used for debugging in `make_predictions.py`

## Information about submissions

### 1. submission

Commit hash: `dc452925594c5838b456a9723844056cd1e6b207`

No data augmentation was done for the first submission.

#### (some) Hyperparameters

- number of epochs = 40
- number of workers = 4
- batch size = 16
- kernel size = 3
- number of hidden layers = 7
- number of hidden units = 20
- number of inputs = 3
- number of outputs = 3
- learning rate = 0.01
- weight decay = 1e-5

#### Final Score

The model scored `23.138` on the challenge server (lower is better).

### 2. submission

Commit hash: `57a17988af44d3afe55274dbd076bafd01a0392b`

Offsets and spacings were selected randomly.
No further data augmentation was done for this submission.

### (some) hyperparameters

- number of epochs = 10
- number of workers = 4
- batch size = 16
- kernel size = 3
- number of hidden layers = 7
- number of hidden units = 20
- number of inputs = 3
- number of outputs = 3
- learning rate = 0.001
- weight decay = 1e-6
- help layers: 2, 3
  - help layers are layers where the known inputs are concatenated with the output of the previous layer.

#### Final Score

The model scored `19.915` on the challenge server (lower is better).

The self-calculated mean loss on the normalized validation set was `0.004835741128772497`.