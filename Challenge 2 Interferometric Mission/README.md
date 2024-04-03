<image src="../images/golomb.png" align="center" alt="Challenge 2: Interferometric Mission"/>

# Challenge 2: Interferometric Mission

### Introduction

Welcome to the GRAIL (Golomb Ruler Advanced Interferometric Lens) project, where our sights are set on one vital mission: high-resolution interferometric measurements. These measurements are crucial to map distortions in space-time in order to effectively navigate through warps and folds. Without their high precision, spacecraft risk being lost in the fabric of space-time, never to return. Scientists figured out that a constellation of satellites can achieve the needed precision by forming a large synthetic aperture. Unbeknownst to them, there is an intricate hurdle that they still need to overcome: how to arrange the satellites such that they maintain a useful configuration over several orbits? It turns out that the configuration of satellites which results in the highest resolution is also the one which best follows the mathematical structure of Golomb patterns. Synchronizing these orbits and coordinating observations requires the collective expertise of scientists, engineers, and innovators like you!

### Theory

To obtain images from distant stars at a high angular resolution, one usually has two options: either one builds one very large telescope with a huge aperture, or one synthetically simulates a large telescope by building multiple smaller telescopes. However this poses a new challenge, in order to avoid redundant information, the smaller telescopes need to be placed carefully following the mathematical structure of Golomb patterns. An optimal Golomb pattern consist of distinct displacements (called baselines) between each telescope on the plane normal to the observation direction, see [Memarsadeghi et al. 2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9815260).

### Description

The constellation of satellites needs to closely follow one satellite called the "mothership". The orbit of this mothership is defined by the following parameters:

* initial conditions (ic) at t = 0
* period

The goal of this challenge is to find initial conditions (relative position and relative velocity with respect to the mothership) of a number of satellites with telescopes, such that the constellation forms the highest amount of distinct baselines at three different times:

* at t = 0
* at t = 1 period of mothership
* at t = 2 periods of mothership

At each of these times, the satellite positions are projected onto three planes: XY, XZ and YZ. The positions are then translated onto a uniform grid of squares centered around the position of the mothership and the amount of unique baselines is computed. (See Tutorial.ipynb)

For this challenge, 3 constellation problems of increasing complexity have to be solved:

* **Beginner**: Mothership in Distance Retrograde Orbit (DRO) with 5 satellites and a grid size of 11x11 squares
* **Intermediate**: Mothership in Distance Retrograde Orbit (DRO) with 40 satellites and a grid size of 21x21 squares
* **Hard**: Mothership in halo orbit with 40 satellites and a grid size of 21x21 squares

### Encoding

The DRO and HALO orbits are both three body periodic orbits in the Eart-Moon system. While not necessary to solve the problem, note that the initial position (km) and velocity (km/s) of the mothership have been normalized using the length unit LU = 389703 and the time unit TU = 382981.

The decision vector $x$, also called *chromosome*, consists of the relative position $dx,dy,dz$ and relative velocity $dvx,dvy,dvz$ with respect to the mothership in cartesian coordinates. For $N$ satellites the solution $x$ needs to contain the initial conditions in the following order:

$x = [dx_1, ... , dx_N, dy_1, ..., dy_N, dz_1, ..., dz_N, dvx_1, ..., dvx_N, dvy_1, ..., dvy_N, dvz_1, ..., dvz_N]$.

Relative positions of $-1$ or $1$ correspond to the edge of the grid where the baselines are computed (the mothership is at the center $(0,0)$ of the grid). Satellites which are placed outside of this range won't appear on the grid and won't be considered when computing the baselines. After each period (so at t = 1 period of mothership and t = 2 periods of mothership), the size of the grid is increased by the factor $inflaction\_factor = 1.23$, thereby slightly relaxing the allowed spread of the satellites.

### Objective

The objective to minimize of this problem is computed as follows:

$f = -\underset{i\in\mathcal{I}}{\max} \underset{s\in\mathcal{S}}{\sum} f_{i,s}$

where $i\in \mathcal{I}=\{1,2,3\}$, $s\in \mathcal{S}=\{\text{XY,XZ,YZ}\}$ and $f_{i,s}$ is the fill factor in the $s$-plane at the $i$-th measurement. Hence:

$f = -\text{max}( [f_{1_{XY}} + f_{1_{XZ}} + f_{1_{YZ}} \quad , \quad f_{2_{XY}} + f_{2_{XZ}} + f_{2_{YZ}} \quad , \quad f_{3_{XY}} + f_{3_{XZ}} + f_{3_{YZ}} ])$

More unique baselines on a given plane result in a larger fill factor, hence the goal is to maintain as many unique baselines on all three planes at the three observation times.

### Submitting

To submit a solution, you can prepare a submission file with the [submission helper](https://api.optimize.esa.int/data/tools/submission_helper.py) via

```python
from submisson_helper import create_submission
create_submission("spoc-3-interferometric-mission","{problem}",x,"submission_file.json","submission_name","submission_description")
```

where `{problem}` is either "tiny-observatory-in-dro",  "large-observatory-in-dro", or "large-observatory-in-halo" (all in lower case), and x is the chromosome. You may then submit the generated JSON file [here](https://optimize.esa.int/submit). This is also demonstrated for the simple problem in the tutorial notebook. After submitting a solution, it will be evaluated immediately.


### Considerations

* If two satellites fall into the same square of the grid they only count as one when computing the baselines
