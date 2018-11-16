# Stability study of deep networks controlling nonlinear ODEs
This project contains all the python code and pickled network data to reproduce the reuslts in the paper:
"On the stability analysis of optimal state feedbacks as represented by deep neural models"

# Bebop rajectories controlled by a Guidance and Control Netowrk (G&CNET)
<p align="center">
  <img align="middle" src="./assets/trajs.png" alt="GECNET neurocontroller for the BEBOP drone" width="500" />
</p>

<p align="center">
  <img align="middle" src="./assets/quad_traj.gif" alt="GECNET controlling the BEBOP drone" width="500" />
</p>

# Linearization of Neurocontrollers
The deep neural networks are trained to imitate the optimal power optimal response and are here loaded from pickled data. The dynamics can then be linearized around the equilibrium point using the network gradient information coming from e.g. backpropagation. Stability analysis and time delayed analysis can thus be performed and, for example, the root locus for the time delay obtained (and thus a stability margin):

<p align="center">
  <img align="middle" src="./assets/locusrootN_3_100.png" alt="Time delay for a GECNET controlling the BEBOP drone" width="300" />
</p>


# High Order Taylor maps
The ODEs describing the neurocontrolled trajectory is then numerically integrated and its result xpanded into a high order Taylor map (note that backpropagation cannot be used here as the derivative order is ~7). The map represents the optimal dynamics close to a nominal trajectory. Convergence radius of the Taylor maps is also studied.

<p align="center">
  <img align="middle" src="./assets/taylormodel.png" alt="Time delay for a GECNET controlling the BEBOP drone" width="300" />
</p>
