
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12654214.svg)](https://doi.org/10.5281/zenodo.12654214)
# statROM

This is Python code to reproduce the results in our reduced order model statFEM paper.

## Overview and Structure
The flowchart Figure 1 in the paper reflects an overview the code as follows:
### Offline phase:
1. Sample parameter space: Implemented in exampleHelmholtz.py as generateParameterSamples()
2. Construct primal/adjoint ROM: Implemented in exampleHelmholtz.py as computeROMbasisSample()
### Online phase:
1. Reduce matrices and solve: Implemented in exampleHelmholtz.py as calcROMprior()
2. Estimate ROM error: This is a core functionality of this code and also implemented within calcROMprior()
3. Data Assimilation: Implemented in exampleHelmholtz.py as getCorrectedROMPosterior()

Both demo files are structured around this framework.
The file hierarchy is structured as follows:
1. demo.py: This is the high level run-file. User parameters are declared and the individual parts of the statROM procedure are called. exampleHelmholtz.py is imported.
2. exampleHelmholtz.py: This file provides the majority of methods necassary for the statROM procedure, including parameter sampling, data generation and data assimilation. It wraps the FEM and ROM solvers, which are provided by RB_solver.py and AORA.py. I talso wraps low-level data assmiliation procedures.
3. RB_solver.py: Here, the FEniCSx FEM solver is implemented along with futher low-level methods for data assimilation. The file also wraps AORA.py to generate the ROM basis.
4. AORA.py: Along with assemble_matrix.py and assemble_matrix_derivative.py, this file implements the AORA procedure.

Details on the implemented classes and methods in the individual files are given in the respective file header.

## Usage
To be able to run the code, FEniCSx (https://fenicsproject.org) is required as a FEM backend. We recommend using the dolfinx Docker image in version 0.6.0. A suitable Dockerfile and VSCode .devcontainer configuration is given. To install the container image in VSCode, follow these steps:

- Make sure that Docker is installed and running on your machine
- Make sure that the devcontainers extension is installed within VSCode
- Open the project folder in VSCode
- Click on "Reopen in Container" when prompted (otherwise press F1 and search for "Reopen in Container")
- The container should now be built automatically according to the information given in the Dockerfile. This can take a while.

 Run
  ```bash
  source dolfinx-complex-mode
  
  ```
in your terminal before trying our examples to enable complex numbers in FEniCSx.


## Demo

Two Demo files are provided:

- demo_1d.py
- demo_2d.py

The first runs the 1D example section of the paper, the second runs the 2D acoustic scattering example.
Simply run
```bash
python3 demo_1d.py

```
to compute results. 
Within the respective file, inside the class 
```python
class UserParameters
```
the most interesting parameters can be changed to observe the effect on the posterior error.

The program will, at the end, output something along
```bash
proposed statROM on ROM prior posterior error:
(0.09747361430260673+0j)
classical statFEM on ROM prior posterior error:
(0.33893244427074987+0j)
classical statFEM on FEM prior posterior error (reference):
(0.007312335634658208+0j)
```
which are the error norms of the proposed method, the classical statFEM on a ROM prior and the statFEM on FEM baseline.
Notice that the output will vary: The forcing, the parameter sample and also the data generation are stochastic. Therefore, there will be randomness also in the posterior solution. 


In the Results folder, in the 1D case an overview plot with priors, posteriors and data will be saved. The desired plots can easily be chosen in the demo script. In the 2D example, a number of .xdmf files, which can be viewed with ParaView, will be saved. The most interesting ones are:
- ROM_error_mean_estimated.xdmf (adjoint GP estimate of the ROM error)
- ROM_error_mean_exact.xdmf (exact ROM error as a reference)
- posteriorFEM_mean.xdmf (Full order reference posterior mean)
- posteriorROM_mean.xdmf (Classical approach posterior mean)
- posteriorCorrectedROM_mean.xdmf (Proposed approach posterior mean)

Please notice that the plots for the parameter studies (Fig. 4-6 and 9) are not created directly by running the example files. These were made by running the code multiple times, compiling the results and constructing the plots individally.

## Support

For support, email l.hermann@tu-braunschweig.de


