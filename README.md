# Bluesim

Bluesim is a realistic three dimensional simulator to test ideas for collective behaviors with Bluebots.

## Branches
- `master`: Used to run experiments on aggregation/dispersion for our Science Robotics publication: https://www.science.org/doi/10.1126/scirobotics.abd8668. 
- `aligning`: Used to run experiments on alignment for our ICRA 2021 publication: https://www.florianberlinger.ch/publications/pdf/berlinger2021evasive.pdf.
- `impressionist`: Used to run experiments on circle formation and flocking.

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

## Additional requirement if animations are desired

- ipyvolume

Installation: Manually following instructions on https://github.com/maartenbreddels/ipyvolume.

## Upload code for an experiment on the virtual Bluebots

*Use the heap implementation for maximum performance! The threads implementation is no longer fully supported.*

1. Go to `*/FastSim/heap/fishfood`

2. Save your new experiment here. If designing a new one, you may start with a copy of `fish_template.py`, which offers some basic functionalities.

## Run an experiment with simulated Bluebots

1. Go to `*/FastSim/heap`

2. Change experimental parameters such as number of fish and simulation time in `simulation.py`.

3. Run `simulation.py` from a terminal, together with the filename of the experiment in `fishfood` you want to simulate, e.g.:

```
python3 simulation.py dispersion
```

Simulation results get saved in `./logfiles/` with a `yymmdd_hhmmss` prefix in the filename. Experimental parameters are saved in `yymmdd_hhmmss_meta.txt`; experimental data in `yymmdd_hhmmss_data.txt`.

Results can be animated by running `animation.py` from a terminal, together with the prefix of the desired file, e.g.:

```
python3 animation.py 201005_111211
```

Animation results get saved as html-files in `./logfiles/` with the corresponding `yymmdd_hhmmss` prefix in the filename. Open with your favorite browser (firefox is recommended for full screen views); sit back and watch the extravaganza!

## Data format
Simulation data in `./logfiles/yymmdd_hhmmss_data.txt` includes the positions and velocities of all fishes (columns) over time (rows) in csv-format of shape:

```
(simulation_time * clock_freq + 1) X (no_fishes * 8),
```

with parameters found in `./logfiles/yymmdd_hhmmss_meta.txt`.

The time interval between rows is `1/clock_freq`. Within a row, the columns contain `no_fishes * 4` positions followed by `no_fishes * 4` velocities. For a given fish, the position are its x-, y-, and z-coordinates and its orientation angle phi; the velocity is the first derivative of the position.

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

## Simulator architecture

The Bluesim simulator has a central database that keeps track of positions, velocities, relative positions, and distances of all simulated robots. The robots are simulated asynchronously and one at a time, ordered by a heap data structure. Each robot has access to a local view of its environment; all robots share the same dynamics. Robot variables such as cognition speed or visual range can be changed, as can the perception complexity by introduction of noise, occlusions, and parsing. The decision making algorithms use the same logic in simulation and on the physical robots. Their syntax looks alike with Python 3 being used everywhere to facilitate simulator-to-robot transitions.

Let me explain the architecture of Bluesim in further detail by going through one simulation step for
one robot:
1. A robot got selected for a simulation step because it had the lowest `time` of all robots in the
heap.
2. The `duration` of the simulation step is drawn as a normal deviate with a mean equivalent
to the expected duration of a single perception-cognition-action cycle (0.5 s), and a standard
deviation of 10% (0.05 s).
4
3. The robot gets its current local view from the central database. This includes either the set of
visible LEDs after occlusions, or the relative positions and distances to visible robots if parsing
is not simulated.
4. Based on this local view, the robot decides on where to move next according to the preprogrammed
behavior and respective algorithms.
5. The dynamics of the robot are simulated for the drawn `duration` according to where the robot
decided to move.
6. The attained new position and velocity is entered in the central database. The respective relative
positions and distances to neighbors are recalculated.
7. The robot re-enters the heap with updated `time = time + duration` (and not necessarily
at the end of the heap, allowing to alter the robot order).
