
# statROM

This is Python code to reproduce the results in our reduced order model statFEM paper.


## Usage
To be able to run the code, FEniCSx (https://fenicsproject.org) is required as a FEM backend. We recommend using the dolfinx Docker image. A suitable Dockerfile and VSCode .devcontainer configuration is given.

Run
```bash
source dolfinx-complex-mode

```
in your terminal before trying our examples to enable complex numbers in FEniCSx.


## Demo

Two Demo files are provided:

- exampleHelmholtz.py
- exampleHelmholtz_scatter.py

The first runs the 1D example section of the paper, the second runs the 2D acoustic scattering example.
Simply run
```bash
python3 exampleHelmholtz.py

```
to compute results.

The script will, at the end, output something along
```bash
adv norm:
((0.011015422140876235+0j), (0.008556106136823139+0j))
easy norm:
((0.013534992654630207+0j), (0.009155798345431258+0j))
fem norm:
((0.010004511934544867+0j), (0.008149144589654142+0j))
```
which are the error norms of the proposed method, the classical statFEM on a ROM prior and the statFEM on FEM baseline.

In the Results folder, in the 1D case an overview plot with priors, posteriors and data will be saved. In the 2D example, a number of .xdmf files, which can be viewd with ParaView, will be saved.

## Support

For support, email l.hermann@tu-braunschweig.de

