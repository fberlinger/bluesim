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

*Use the heap implementation for maximum performance! The threads implementation is not currently fully supported.*

1. Go to `*/FastSim/heap/fishfood`

2. Save your new experiment here. If designing a new one, you may start with a copy of `fish_template.py`, which offers some basic functionalities.

## Run an Experiment with Simulated Bluebots

1. Go to `*/FastSim/heap`

2. Change experimental parameters such as number of fish and simulation time in `simulation.py`.

3. Run `simulation.py` from a terminal, together with the filename of the experiment in `fishfood` you want to simulate, e.g.:

```
python simulation.py dispersion
```

Simulation results get saved in `./logfiles/` with a `yymmdd_hhmmss` prefix in the filename. Experimental parameters are saved in `yymmdd_hhmmss_meta.txt`; experimental data in `yymmdd_hhmmss_data.txt`.

Results can be animated by running `animation.py` from a terminal, together with the prefix of the desired file, e.g.:

```
python animation.py 201005_111211
```

Animation results get saved as html-files in `./logfiles/` with the corresponding `yymmdd_hhmmss` prefix in the filename. Open with your favorite browser (firefox is recommended for full screen views); sit back and watch the extravaganza!

## Data Format
Simulation data in `./logfiles/yymmdd_hhmmss_data.txt` includes the positions and velocities of all fishes (columns) over time (rows) in csv-format of shape:

```
(simulation_time * clock_freq + 1) X (no_fishes * 8),
```

with parameters found in `./logfiles/yymmdd_hhmmss_meta.txt`.

The time interval between rows is `1/clock_freq`. Within a row, the columns contain `no_fishes * 4` positions followed by `no_fishes * 4` velocities. For a given fish, the position are its x-, y-, and z-coordinates and its orientation angle phi, the velocity is the first derivative of the position.

Data is easily loaded into matrix format with numpy loadtxt, e.g.:

```
data = np.loadtxt('./logfiles/yymmdd_hhmmss_data.txt', delimiter=','),
```

and can be sliced for looking at a particular fish `i`, or instance in time `t` as follows:

```
pos_i = data[:, 4*i : 4*i+4]
vel_i = data[:, 4*no_fishes+4*i : 4*no_fishes+4*i+4]

pos_t = data[t, :no_fishes*4]
vel_t = data[t, no_fishes*4:]
```

## Simulation Architecture
tbd: explain environment, explain heap, (discuss threads)
