# BlueSim

BlueSim is a realistic three dimensional simulator to test ideas for collective behaviors with BlueBots.

## Requirements

- Python 3.6
- Jupyter 1.0
- Numpy
- Scipy
- Matplotlib
- (PIP _not mandatory but recommended_)

## Installation

Either install Jupyter, Numpy, Scipy, and Matplotlib via PIP:

```
git clone https://code.harvard.edu/flb979/FISH && cd FISH
pip install -r ./requirements.txt
```

Or manually via https://jupyter.org/install and https://scipy.org/install.html

## Additional Requirement if Animations are Desired

- ipyvolume

Installation: Manually following instructions on https://github.com/maartenbreddels/ipyvolume.

## Upload Code for an Experiment on the Virtual BlueBots

Go to the subfolder `fishfood`, choose one of the following experiments, and copy its file ending in `.py` to the current `BlueSim` folder:

- `blindspot.py`: An investigation on how the blind spot behind its own body affects BlueBot's dispersion.
- `blockingsphere.py`: An investigation on how the blocking sphere surrounding its own body affects BlueBot's aggregation.
- `orbit.py`: A single robot orbits around a fixed center.
- `millingabout.py`: Several robots orbit around a fixed center.
- `waltz.py`: Two robots orbit around each other.

Rename the copied file in `BlueSim` to `fish.py`.

## Run an Experiment with Simulated BlueBots

BlueSim will run your selected experiment that you copied and renamed to `fish.py`.

Open the jupyter notebook:

```
jupyter notebook
```

and within that notebook open the file ending in `.ipynb` corresponding to your chosen experiment.

You may change experimental parameters such as number of fish in the notebook. You could even change control code directly in `fish.py`and create your own experiment. However, doing that, you would also have to deal with your own bugs. You should rather not touch any other source code for `BlueSim`. Unexpected errors might occur.

Please run each cell in the notebook individually! **Warning**: Using `Run All` will not work
as the experiments start several threads for every fish and the code execution
async, hence, Jupyter Notebook runs new cells to quickly before others finished.

Do not run any cells generating the animation if you have not installed ipyvolume.

Sit back and watch the extravaganza!

<!---
## Run

Open the jupyter notebook:

```
jupyter notebook
```

and within that notebook open one of the following experiment files ending in `.ipynb`:

- `millingabout.ipynb`
-->
