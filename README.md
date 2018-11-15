# Stability study of deep networks controlling nonlinear ODEs
This repo contains all the python code and data to reproduce the reuslts in the paper:
"On the stability analysis of optimal state feedbacks as represented by deep neural models"

# Some cool trajectories controlled by a deep network
<p align="center">
  <img align="middle" src="./assets/trajs.png" alt="GECNET neurocontroller for the BEBOP drone" width="500" />
</p>

<p align="center">
  <img align="middle" src="./assets/quad_traj.gif" alt="GECNET controlling the BEBOP drone" width="500" />
</p>

# Linearization of Neurocontrollers
The deep neural networks are trained to imitate the optimal power optimal response and are here loaded from pickled data and the dynamics linearized around the equilibrium point. Stability analysis and time delayed analysis can thus be performed:

<p align="center">
  <img align="middle" src="./assets/locusrootN_3_100.png" alt="Time delay for a GECNET controlling the BEBOP drone" width="300" />
</p>


# High Order Taylor maps
The ODEs describing the neurocontrolled trajectory is numerically integrated and the final achieved point expanded in high order Taylor map (that is its gradients and more are computed with respect to parameters). The map represents the optimal dynamics close to a nominal trajectory. Convergence radius of the Taylor maps is also studied:

<p align="center">
  <img align="middle" src="./assets/taylormodel.png" alt="Time delay for a GECNET controlling the BEBOP drone" width="300" />
</p>
