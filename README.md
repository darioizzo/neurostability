# Stability study of deep networks controlling nonlinear ODEs
This project contains all the python code and pickled network data to reproduce the reuslts in the paper:

Dario Izzo, Dharmesh Tailor and Thomas Vasileiou: "On the stability analysis of optimal state feedbacks as represented by deep neural models"

# Guidance and Control Network (G&CNET)
**Definition:**: A G&CNET (Guidance and Control Network) is a type of neuro controller, in particular a deep, artificial, feed forward neural network trained on state-action pairs representing the optimal control actions, with respect to the performance index:
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=J&space;=&space;\int_{t_0}^{t_f}&space;l(\mathbf&space;x,&space;\mathbf&space;u,&space;t)&space;dt" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J&space;=&space;\int_{t_0}^{t_f}&space;l(\mathbf&space;x,&space;\mathbf&space;u,&space;t)&space;dt" title="J = \int_{t_0}^{t_f} l(\mathbf x, \mathbf u, t) dt" /></a>
</p>
In other words a G&CNET is a **neural optimal feedback** for a non linear dynamical system.

An example for a GECNET architecture is given below, note the **softplus units** giving continuity properties to the resulting actions.

<p align="center">
  <img align="middle" src="./assets/gecnet.png" alt="GECNET neurocontroller for the BEBOP drone" width="500" />
</p>

For example a GECNET can perform **real time optimal manouvre on board** of a BEBOP Parrot drone:

<p align="center">
  <img align="middle" src="./assets/trajs.png" alt="GECNET neurocontroller for the BEBOP drone" width="500" />
</p>

<p align="center">
  <img align="middle" src="./assets/quad_traj.gif" alt="GECNET controlling the BEBOP drone" width="500" />
</p>

# Stability of G&CNETS Neurocontrollers
These deep neural networks are trained to imitate the optimal response and, in the notebooks here available, are loaded from pickled data. The system dynamics can then be linearized around an equilibrium point (hovering, in the case of a BEBOP drone) using the network gradient information coming from e.g. backpropagation. **Stability and time delayed analysis can thus be performed** and, for example, the root locus for the time delay obtained (and thus a stability margin):

<p align="center">
  <img align="middle" src="./assets/locusrootN_3_100.png" alt="Time delay for a GECNET controlling the BEBOP drone" width="300" />
</p>


# Stability of a G&CNET nominal trajectory
The ODEs describing the neurocontrolled trajectory is then numerically integrated and its result expanded into a high order Taylor map (note that backpropagation cannot be used here as the derivative order is ~7, so differential algebraic techniques are used instead). **The map represents the optimal dynamics close to a nominal trajectory** and can thus be used to study and prove its stability. Convergence radius of the Taylor maps is also studied.

<p align="center">
  <img align="middle" src="./assets/taylormodel.png" alt="Time delay for a GECNET controlling the BEBOP drone" width="500" />
</p>
