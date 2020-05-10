# FastSim

FastSim is a realistic three dimensional simulator to test ideas for collective behaviors with Bluebots. Built on BlueSim, it is a faster but less decentralized implementation.

## Requirements

- Python 3.6
- Matplotlib
- Numpy
- Scipy
- (PIP _not mandatory but recommended_)

## Installation

Either install Matplotlib, Numpy, and Scipy via PIP:

```
git clone https://code.harvard.edu/flb979/FastSim && cd FastSim
pip install -r ./requirements.txt
```

Or manually via https://scipy.org/install.html

## Additional Requirement if Animations are Desired

- ipyvolume

Installation: Manually following instructions on https://github.com/maartenbreddels/ipyvolume.

## Upload Code for an Experiment on the Virtual Bluebots

*Use the heap implementation for maximum performance! The thread implementation is not currently fully supported.*

1. Go to `*/FastSim/heap`

2. Delete `fish.py`

3. Go to the subfolder `fishfood`, create a copy of `fish_template.py` and rename it, implement your Bluebot code there; **or** choose an existing experiment-file

4. Copy your file to the `heap` parent-folder, and rename it to `fish.py`

**Warning: Any changes made directly in `fish.py` will be lost during the next execution of step 2. Save your final code in the `fishfood` folder.**

## Run an Experiment with Simulated Bluebots

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
