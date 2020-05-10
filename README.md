# FastSim

FastSim is a realistic three dimensional simulator to test ideas for collective behaviors with Bluebots. Built on BlueSim, it is a faster but less decentralized implementation.

## Requirements

- Python 3.6
- Numpy
- Scipy
- Matplotlib
- json
- (PIP _not mandatory but recommended_)

## Installation

Either install Jupyter, Numpy, Scipy, Matplotlib, and json via PIP:

```
git clone https://code.harvard.edu/flb979/FISH && cd FISH
pip install -r ./requirements.txt
```

Or manually via https://jupyter.org/install and https://scipy.org/install.html

## Additional Requirement if Animations are Desired

- ipyvolume

Installation: Manually following instructions on https://github.com/maartenbreddels/ipyvolume.

## Upload Code for an Experiment on the Virtual Bluebots

**Use the heap implementation for maximum performance!**

Go to the subfolder `fishfood`, choose one of the following experiments, and copy its file ending in `.py` to the current `FastSim` folder:

Rename the copied file in `BlueSim` to `fish.py`.

## Run an Experiment with Simulated Bluebots

FastSim will run your selected experiment that you copied and renamed to `fish.py`.

Change experimental parameters such as number of fish and simulation time in `main.py`.

Run `main.py` from a terminal, together with an experiment description, e.g.:

```
python main.py schooling
```

Simulation results get saved in `./logfiles/` with a `yymmdd_hhmmss` prefix in the filename. Experimental parameters are saved in `yymmdd_hhmmss_meta.txt`; data in `yymmdd_hhmmss_data.txt`.

Results can be animated by running `animation.py` from a terminal, together with the prefix of the desired file, e.g.:

```
python animation.py 201005_111211
```

Animation results get saved as html-files in `./logfiles/` with the corresponding `yymmdd_hhmmss` prefix in the filename. Open with your favorite browser (firefox is recommended for full screen views); sit back and watch the extravaganza!

## Data Format
