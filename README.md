
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12654214.svg)](https://doi.org/10.5281/zenodo.12654214)
# statROM

This is Python code to reproduce the results in our reduced order model statFEM paper.


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


In the Results folder, in the 1D case an overview plot with priors, posteriors and data will be saved. The desired plots can easily be chosen in the demo script. In the 2D example, a number of .xdmf files, which can be viewd with ParaView, will be saved. The most interesting ones are:
- ROM_error_mean_estimated.xdmf (adjoint GP estimate of the ROM error)
- ROM_error_mean_exact.xdmf (exact ROM error as a reference)
- posteriorFEM_mean.xdmf (Full order reference posterior mean)
- posteriorROM_mean.xdmf (Classical approach posterior mean)
- posteriorCorrectedROM_mean.xdmf (Proposed approach posterior mean)

## Support

For support, email l.hermann@tu-braunschweig.de


