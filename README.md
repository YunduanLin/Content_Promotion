# Code for the paper -- Content Promotion for Online Content Platforms with the Diffusion Effect

Here the authors provide the code used in the paper, [Content Promotion for Online Content Platforms with the Diffusion Effect]<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3863104>. The authors are not allowed to reveal any datasets, as required by the platform which they collaborated with. You could contact the authors if you have any questions.

## Files

---

### Estimation

estimation.ipynb

- This file records the code for parameter estimation, including the OLS method for the BDM and the P-BDM as well as the D-OLS method for the P-BDM method, as described in Section 5 and partial results in Section 6.

### Simulation

simplatform.py

- This file includes a class that is used to establish the simulation environment for the online content platform, including the external environment dynamics, promotion operations, and the adoption process of all agents. The basic settings can be found in Section 6.3.1.

CGPO.py

- This file includes a class that is used to solve the (content generation and) promotion optimization problem. The resolution of the problem depends on the commercial solve Gurobi as well as the approximation algorithm we discussed in Section 4.

test.ipynb

- This is an example test instance. By running this code, we are able to store the simulation results in Folder CGPO_L1_C2.

