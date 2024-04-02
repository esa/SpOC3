<image src="../images/cubes.png" align="center" alt="Challenge 3: Programmable Cubes"/>

# Challenge 3: Programmable Cubes

### Introduction

Welcome to the OASIS project, one of humanity's flagship research projects for in-space assembly of megastructures! [Even way back in the 21st century](https://www.media.mit.edu/projects/tesserae-self-assembling-space-architecture/overview/), scientists envisioned ships and habitats to self-assemble in space from small components that can be easily deployed from Earth. Now this dream is finally getting closer with the programmable cubes technology: small cubes of different designs capable of forming complex and fully functional structures when interconnected.

The missing building block for this technology is an efficient and scalable algorithm for assembling structures. To solve this, we need the help of every student, scientist and engineer out there interested in contributing to the OASIS project! To test your algorithms, we generated prototype scenarios where you have to assemble two historical spacecraft from the early days of human space exploration: **The International Space Station (ISS)** and the **James Webb Space Telescope (JWST)**. For final evaluation, we also give you a single assembly scenario of the planned multi-generational interstellar ship **Enterprise**. If successful, your algorithm might give us the missing puzzle piece for constructing this massive ship in orbit and boldly go where no one has gone before! 

### Description

The goal of this challenge is to program an ensemble of reconfigurable cubes to morph as accurately and as fast as possible into target space structures. The structures are characterised by the number of cubes they are composed of, the types of cubes they are made of (representing different types of, e.g., materials or functionality), and the locations in 3D space that have to be occupied by a cube of a given type to successfully construct the structure. Cubes that share the same type are interchangable when building a structure.

For this challenge, three assembly problems of increasing size and complexity have to be solved:

* assembling the International Space Station (ISS): 148 cubes, 3 cube types.
* assembling the James Webb Space Telescope (JWST): 643 cubes, 6 cube types.
* assembling the USS Enterprise: 1472 cubes, 10 cube types.

For each problem, an initial position for cubes of each type are provided. From this initial configuration, the ensemble can morph its shape using "pivoting maneuvers", i.e., cubes can pivot around their edges to change their position within the ensemble (see next section for details). Your goal is to find the shortest sequence of commands, i.e., selecting cubes and pivoting them, to construct the target structure as best as possible starting from the initial configuration.

### Moving Cubes

Cubes can perform two types of moves: (i) pivoting around a single cube, i.e., connecting to another face of the same cube (called "Pivoting" in Fig. 1), and (ii) pivoting along a cube, i.e., moving from one cube to another (called "Traversal" in Fig. 1). 

![Pivoting Cubes](https://hcie.csail.mit.edu/research/Electrovoxel/images/ryo-concept-cam.png)
***Fig. 1:*** *Illustration of the pivoting maneuver. Taken from [Nisser et al. 2022](https://hcie.csail.mit.edu/research/Electrovoxel/electrovoxel.html) ([DOI](https://doi.org/10.1109/ICRA46639.2022.9811746)).*

In this challenge, only the axis of rotation and the rotation direction (clockwise or counter-clockwise) has to be provided. Whether a "Traversal" or "Pivoting" (terminology from Fig. 1) move is applied is automatically determined by our code. Thus, we define the following commands:

* `0` and `1`: counter-clockwise and clockwise rotation in x-y plane (around z-axis).
* `2` and `3`: counter-clockwise and clockwise rotation in y-z plane (around x-axis).
* `4` and `5`: counter-clockwise and clockwise rotation in x-z plane (around y-axis).

Two criteria have to be satisfied for a cube to perform a given maneuver:

* first, removing the cube from the ensemble is not allowed to split it into two disconnected components, i.e., the whole ensemble has to stay connected.
* Second, the cube is not allowed to collide with other cubes while moving, and it has to be able to connect to another cube of the ensemble when arriving at its final position. 

The whole logic of checking these conditions and updating the state of the cube ensemble accordingly is implemented in the class `ProgrammableCubes` found in the provided UDPs.
The logic for moves (i.e., determining the final position of a pivoting cube and checking for collisions) is set up in [CubeMoveset.py](https://api.optimize.esa.int/data/spoc3/cubes/CubeMoveset.py).
For more details and an illustration, see, e.g., [Sung et al. 2015](https://doi.org/10.1109/ICRA.2015.7139451) (Figure 1) and [Akitaya et al. 2021](https://doi.org/10.1007/s00453-020-00784-6) (Figure 1). 

### Data

You are provided with cube positions and types of both the initial and target configuration for each problem ([data/ISS](https://api.optimize.esa.int/data/spoc3/cubes/data/ISS), [data/JWST](https://api.optimize.esa.int/data/spoc3/cubes/data/JWST/), [data/Enterprise](https://api.optimize.esa.int/data/spoc3/cubes/data/Enterprise/)). The paths to the data are set in the json file of each problem [cubes/problems/](https://api.optimize.esa.int/data/spoc3/cubes/problems/). **By default**, these are set to `.\data\spoc3\cubes\data\{problem name}` and **have to be changed if the folder locations vary on your machine**. The path to the json files is set in the python file which contains the UDP (`folder_path` variable).

Moreover, we provide four tutorial [IPython Notebooks](https://github.com/esa/SpOC3/tree/main/Challenge%203%20Programmable%20Cubes) illustrating the basic functionality of the evaluation code.

All files and data can be conveniently found on the accompanying [GitHub repository](https://github.com/esa/SpOC3). If you use the files from github, data paths will already be set to be consistent with the repository structure assuming code is executed from within the `Challenge 3 Programmable Cubes/` folder.

### Encoding

The decision vector $x$, also called *chromosome*, consists of a sequence of cube indices (`-1` to `number of cubes - 1`) and move commands (`0` to `5`).
The cube indices correspond to the elements in the list of initial cube positions that we provide. The entry `-1` is not a cube index but signals the end of the chromosome, allowing early termination of the reconfiguration process (i.e., everything that follows the first occurance of `-1` in the chromosome is ignored during fitness evaluation).

Thus, a chromosome allowing for a maximum of N+1 cube moves looks as follows:

$x = [c_0, m_0, c_1, m_1, ... , c_N, m_N, -1] ,$

with $c_i$ and $m_i$ being the cube index and command index used in the $i^\mathrm{th}$ step, respectively. The chromosome always has to end with `-1` to ensure proper fitness evaluation on the optimize platform. The entry `-1` can appear multiple times in the chromosome, but fitness evaluation will be terminated after encountering the first of these entries.
Note that the length of the effective chromosome (until the first `-1` is encountered) goes into the fitness as well.

For the ISS problem, a maximum of 6,000 commands is possible. For the JWST problem, a maximum of 30,000 commands is possible. For the Enterprise, a maximum of 100,000 commands is possible.

### Objective

The main objective of this challenge is to maximise the overlap of cube positions to the target configuration. As a secondary objective, this has to be done using the least amount of commands possible.

Given a chromosome $x$, we define $C(x)$ to be the number of cubes that are in the correct location after applying the chromosome to an ensemble starting from the initial cube positions.
To calculate this quantity, we simply check for each cube of a given type in the target configuration whether a cube of the same type occupies this location in the obtained configuration of the solution.
In addition, we define $L(x)$ to be the number of commands issued (i.e., the number of chromosome entries until `-1` is encountered), $L_\mathrm{max}$ is the maximum number of possible commands (i.e., the total chromosome length), and $C_\mathrm{max}$ the number of cubes in the ensemble.

The fitness of chromosome $x$ is then given by

$f(x) = - \frac{F(x) - F_0}{1-F_0}  ,$

with

$F(x) = (1-\alpha) \cdot \frac{C(x)}{C_\mathrm{max}} + \alpha \cdot \left(1 - \frac{L(x)}{L_\mathrm{max}} \right)  ,$

and $\alpha = 0.1$. $F_0$ is the value of $F(\cdot)$ obtained when applying no maneuvers, i.e., when applying a chromosome $x_0$ with $L(x_0) = 0$, and thus $F_0 = F(x_0)$. 
Hence, when leaving the cube ensemble unchanged from the initialisation, the fitness is $0$, $f(x_0) = 0$.
A solution worse than the initialisation has positive fitness, with the worst fitness being $\frac{F_0}{1-F_0}$.
The theoretically best value of the fitness function is $-1$.

For any decision vector $x$, the fitness can be retrieved with the following line of code:

```
fitn = udp.fitness(x)
```

### Submitting

To submit a solution, you can prepare a submission file with the [submission helper](https://api.optimize.esa.int/data/tools/submission_helper.py) via

```python
from submisson_helper import create_submission
create_submission("spoc-3-programmable-cubes","{problem}",x,"submission_file.json","submission_name","submission_description")
```

where `{problem}` is either `iss`, `jwst`, or `enterprise` (all in lower case), and [submit it](https://optimize.esa.int/submit). This is demonstrated for the three problems in the fourth tutorial IPython notebook. After submitting a solution, it will be evaluated immediately. For this challenge, **this process can take a bit of time, so please be patient** and wait until the fitness of your submitted solution is returned.

### Utilities / Hints

* We will be using [GitHub](https://github.com/esa/SpOC3) as our hub for communication with competitors. Our primary means of communication will be the `Discussions` feature on our repository.

* Have a detailed look at the [User Defined Problem (UDP)](https://optimize.esa.int/challenge/spoc-3-programmable-cubes/ISS) accessible under `Evaluation code` to understand how the evaluation works. The UDPs of all three problems are identical - with the only difference being the data and parameters loaded to set up the problems.

* You can set the file path to the data on your system in the json files found in the folder `problems`. **You have to** set the path (`folder_path`) to the json files in `programmable_cubes_UDP.py` for the code to work on your machine.

* We provide four tutorial IPython notebooks that illustrate the main functionality of the code and include example solutions, which can be found in our [Github repository](https://github.com/esa/SpOC3/tree/main/Challenge%203%20Programmable%20Cubes).

* To create a problem instance for each problem, simply create an UDP object with the problem name: `udp = programmable_cubes_UDP('ISS')`, `udp = programmable_cubes_UDP('JWST')`, `udp = programmable_cubes_UDP('Enterprise')`.

* Plots of the cube ensemble configuration can be looked at using `udp.plot(which_one, cube_type_to_plot)`. By setting `which_one="ensemble"`, you plot the final configuration obtained after applying the chromosome (fitness has to be called beforehand). By setting `which_one="target"`, the target configuration will be plotted. `cube_type_to_plot` is a list containing the indices of the cube types to be plotted, e.g., `cube_type_to_plot=[0,5]`. In addition, there are two optional arguments for plotting custom configurations and custom cube types, `custom_config` and `custom_cube_types`.

* By calling `udp.pretty(x)`, illegal moves (and why they are illegal) will be printed to the terminal during fitness evaluation.

* Start with the simpler challenge (ISS) to develop a method for tackling the harder ones (JWST and Enterprise).

* Keep in mind that later parts of the chromosome depend on earlier parts (due to the cube ensemble configuration changing).

* You can terminate chromosome evaluation early by using `-1` as a cube index. Thus, you do not have to use all entries available in the chromosome (in fact, shorter solutions yield better fitness).

* You can start from intermediate configurations by giving them as an argument to the fitness function, i.e.,  `udp.fitness(x, initial_configuration = your_configuration)`. This way, you do not have to evaluate the whole chromosome all the time and start from later stages instead (keeping initial parts of the chromosome constant). Partial evaluation of the chromosome also considerably speeds up fitness evaluation.

### Acknowledgments

* For the ISS, we adapted the 3D model ["International Space Station - ISS"](https://skfb.ly/6VQvn) by Osvaldo Mendes licensed under [Creative Commons Attribution](http://creativecommons.org/licenses/by/4.0/).
* For the JWST, we adapted the 3D model ["JWST (james webb space telescope)"](https://skfb.ly/o8yLN) by Paul (Sketchfab) licensed under [Creative Commons Attribution](http://creativecommons.org/licenses/by/4.0/).
* For the Enterprise, we adapted the 3D model ["U.S.S. Enterprise NCC-1701"](https://skfb.ly/o6p7Z) by riker446 licensed under [Creative Commons Attribution](http://creativecommons.org/licenses/by/4.0/).
* To convert files into .obj format, we used the free software [Blender](https://www.blender.org/).
* To voxelize and adapt the 3D models, we used the free software [MagicaVoxel](https://ephtracy.github.io/).
* To load .vox files into Python, we used the open source package [py-vox-io](https://github.com/gromgull/py-vox-io) by gromgull.
