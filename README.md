For the third time, the European Space Agency's Advanced Concepts Team
(ACT) presents in cooperation with the [The Genetic and Evolutionary
Computation Conference
(GECCO)](https://gecco-2024.sigevo.org/HomePage) the *Space
Optimization Competition (SpOC)*. Look forward to three challenging
problems placed in an futuristic space mission scenario.

<Image src="images/spoc-banner.png" align="center" alt="SpOC 2023"/>

# SpOC 3: Orbital Megastructures 

The year is 2287 and you wander through the tunnels of Lunar
City. Passing by you favorite kiosk, which is the only one on the Moon
that still sells newspapers made out of *real* paper, an article
in *The Earth Observer* catches your eyes. You grab a copy and a cup of
100% Quetelet-Crate-Bean coffee.

**Winners of the Orbital Megastructure competition announced.**

After decades of negotiation, two of the most ambitious projects that
humanity has ever conceived are finally scheduled to commence next
year. The Golomb Ruler Advanced Interferometric Lens (GRAIL) and the
Orbital Assembly of Self-organising Interstellar Spacecraft (OASIS)
projects have been selected among more than three dozen candidates
competing for the immense amounts of funding. Despite the fierce
competition, the unanimous opinion is that the sheer ambition and
scope of cooperation on these projects will elevate humanity to a new
level as a species. 

Although the winning projects could not be more different, they are
inexorably linked. The veteran GRAIL project (the oldest of all
proposals, first conceived in the distant 21st century), overcame
severe criticism and convinced the judges with solid arguments for its
usefulness. The GRAIL project became relevant overnight with the
discovery of Gaia Gemina - the likely habitable 'Earth twin' planet
located squarely in the middle of the Goldilocks zone around an
inconspicuous Sun-like star in the constellation of Sagittarius. The
GRAIL team promises that the project will be completed within record
time using new materials and orbital stabilisation mechanisms,
enabling astronomers to examine Gaia Gemina and its cosmic
neighbourhood in unprecedented detail just a few years from now. 

In contrast, OASIS is a latecomer to the project race. It represents
the hope of humanity to become a space-faring species by sending the
first crew of people to Gaia Gemina. Initially viewed as a practically
impossible task, the construction of the multi-generational
interstellar ship is now scheduled to begin in orbit around Earth at
the same time as the GRAIL project. The ship - which has yet to be
given a name - must be large and sophisticated enough to sustain
several generations of pioneers on their way to Gaia Gemina. 

Of course, both GRAIL and OASIS would depend on the outcome of the
Graph Reduction Algorithm for Planetary-scale Hyperoptimisation
(GRAPH) meta-project. Originally selected as the tool of choice for
solving challenges such as those posed by GRAIL and OASIS, the
pressure for this approach to work has increased manyfold with the
completion of the multi-year negotiation marathon for funding GRAIL
and OASIS. As the winning projects are scheduled to commence next
year, it remains to be seen whether GRAPH was indeed the right choice. 

Whatever the outcome, rallying behind these projects - not at the
local or national but at the _planetary_ level - is no small feat, and
the sense of accomplishment and anticipation is already
palpable. Needless to say, the stakes are high for all projects to
succeed in the mission to guarantee the safety of the first intrepid
interstellar explorers! 

## Competition Structure

SpOC 3.0 contains three distinct problems centered around a futuristic
space mission. Starting from **1 April 2024, End of Day, Anywhere on Earth (AoE)** you have **three** months
to tackle these challenges to secure a spot on the leaderboard, i.e., until
**30 June 2024, End of Day, Anywhere on Earth (AoE)**.

**Detailed technical descriptions for the three challenges to be
solved will be made available on the [Optimise
platform](https://optimise.esa.int) from the same date.** 


### Challenge 1: Torso Decompositions

<image src="images/torso.png" align="right"  alt="Torso Decomposition" width="300" height="200"/>

Before we can start designing astonishing megastructures in the GRAIL
or OASIS project, we need to arrange the building blocks
suitably. These blocks, called nodes or vertices, are bound to
specific pairwise interactions. Some blocks clearly belong to the same
part of the megastructure, while others share no interaction. For an
effective and energy-efficient structure setup, it is critical to
identify an ordering of the nodes that indicates parts of the structure that can be
assembled concurrently, and that encodes a global blueprint for recursively
constructing the megastructure. At the Graph Reduction Algorithm for
Planetary-scale Hyper-optimisation (GRAPH) meta-project, we develop
the toolkit for exactly this task -- and we hope that engaged scientists and
engineers like you will help us to complete the project in time!


### Challenge 2: Interferometric Mission

<image src="images/golomb.png" align="right"  alt="Interferometric Mission" width="300" height="200"/>

Welcome to the GRAIl project, where our sights are set on one vital
mission: high-resolution interferometric measurements. These
measurements are crucial to map distortions in space-time in order to
effectively navigate through warps and folds. Without their high
precision, spacecraft risk being lost in the fabric of space-time,
never to return. Scientists figured out that a constellation of
satellites can achieve the needed precision by forming a large
synthetic aperture. Unbeknownst to them, there is an intricate hurdle
that they still need to overcome: how to arrange the satellites such
that they maintain a useful configuration over several orbits? It
turns out that the configuration of satellites which results in the
highest resolution is also the one which best follows the mathematical
structure of Golomb patterns. Synchronizing these orbits and
coordinating observations requires the collective expertise of
scientists, engineers, and innovators like you!


### Challenge 3: Programmable Cubes

<image src="images/cubes.png" align="right"  alt="Programmable Cubes" width="300" height="200"/>

Welcome to the OASIS project, one of humanity's flagship research
projects for in-space assembly of megastructures! [Even way back in
the 21st century](https://www.media.mit.edu/projects/tesserae-self-assembling-space-architecture/overview/),
scientists envisioned ships and habitats to self-assemble in space
from small components that can be easily deployed from Earth. Now this
dream is finally getting closer with the programmable cubes
technology: small cubes of different designs capable of forming
complex and fully functional structures when interconnected.

The missing building block for this technology is an efficient and
scalable algorithm for assembling structures. To solve this, we need
the help of every student, scientist and engineer out there interested
in contributing to the OASIS project! To test your algorithms, we
generated prototype scenarios where you have to assemble two
historical spacecraft from the early days of human space exploration:
**The International Space Station (ISS)** and the **James Webb Space
Telescope (JWST)**. For final evaluation, we also give you a single
assembly scenario of the planned multi-generational interstellar ship
**Enterprise**. If successful, your algorithm might give us the
missing puzzle piece for constructing this massive ship in orbit and
boldly go where no one has gone before!

### Guidelines and Rules

The competition will be hosted on the
  [Optimise](https://optimise.esa.int/) platform developed by the
  Advanced Concepts Team. Participants will need to register online on the
  platform, and solution entries will need to be submitted via
  Optimise for validation. While SpOC is organized in cooperation with
  GECCO 2024, it is *not* required to attend GECCO 2024 in order to
  participate in SpOC.

- Your objective is to propose and implement metaheuristic algorithms
  to solve the proposed optimisation challenges. 
- In order to validate your solutions, we will provide you with Python
  validation code for each of the three challenges. This code includes
  problem definitions in the [Pygmo](https://esa.github.io/pygmo2/#)
  user-defined problem (UDP) format, examples of solutions, and
  visualisation tools. 
- You have until **30 June 2024** to submit your entries via the dedicated portal [Optimise](https://optimise.esa.int/).
- Please comply with our [basic code of
  honour](https://optimise.esa.int/terms). The ACT reserves the right
  to exclude users from the competition if they abuse the evaluation
  system.

## Scoring and Winner Selection 

This year, SpOC implements a 2-out-of-3 scoring scheme that, simply
put, means that for the overall competition only two 
challenges are taken into account. In detail, you will obtain a *local score*
$s_i$ for each of the three challenges computed via the the rules below,
then your *global score* is $g=\max(s_1+s_2, s_1+s_3, s_2+s_3)$. The global score defines your place on the overall SpOC
leaderboard and the winners of SpOC 3. 

Each challenge $i=1,2,3$ contains *three problems* (easy, medium, and
hard). Every problem has its own leaderboard (visible on
[Optimise](https://optimise.esa.int/)) that ranks participants
according to the objective of the challenge. The top ten ranks on the
leaderboard of the easy instance get $e_i=10,9,8,\dots,1$ points, similarly
$m_i=\frac{4}{3}10,\frac{4}{3}9,\dots,\frac{4}{3}$ points for the top
ten teams on the medium problem, and 
$h_i=(\frac{4}{3})^2 10,(\frac{4}{3})^2 9,\dots,(\frac{4}{3})^2$ points for the hard
problem. The score $s_i$ for the challenge then simply is the sum $s_i=e_i+m_i+h_i$.

We wish all participants the best of luck and are excited to see what you accomplish!

## Timeline

<Image src="images/timeline.png" alt="timeline" />

The Space Optimization Competition starts at the **first of April, End
of Day, Anywhere on Earth (AoE):** 

<Image src="http://i.countdownmail.com/36faif.gif" border="0" alt="countdownmail.com" align="center"/>

**NOTE**: The submission portal remains open after **30 June
2024, End of Day, Anywhere on Earth (AoE)**. Submissions received after that date will not be taken into
consideration for the competition, but still appear on the
leaderboard. 


## Contact

Our primary means of communication with competitors will be the
*Discussions* feature on this repository. Please use it to ask
any questions you may have about the challenges or to exchange
information with us. We will do our best to respond to your questions
in a timely manner. 

If you encounter a bug in the code, please use this repository's
*Issues* feature to report it. 
