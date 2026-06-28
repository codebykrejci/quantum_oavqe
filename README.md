
# Quantum band structure calculation using OAVQE and Constant Measurement Protocol

## Introduction
This repository contains the code for “Minimum Measurements Quantum Protocol for Band Structure Calculations”, available at https://arxiv.org/pdf/2511.04389
. The code implements the **Orthogonal Ansatz Variational Quantum Eigensolver (OA-VQE)** method and the **Constant Measurement Protocol** to calculate the electronic band structure of several materials using the tight-binding model. The calculation results are stored in an **HDF5** file, and a visualisation script enables comparison between the OAVQE results and classical exact diagonalisation.

## Requirements and Installation

To run and use this project, you need to have **Python** installed along with the packages listed in the `requirements.txt` file.
The core dependencies are:

- **Qiskit** — quantum circuits and primitives
- **SciPy** — optimisation algorithms

Below is the full list of required packages as specified in the `requirements.txt` file:

```txt
annotated-types==0.7.0
certifi==2026.6.17
cffi==2.0.0
charset-normalizer==3.4.7
contourpy==1.3.3
cryptography==49.0.0
cycler==0.12.1
dill==0.4.1
fonttools==4.63.0
h5py==3.16.0
ibm-cloud-sdk-core==3.25.0
ibm-platform-services==0.76.0
ibm-quantum-schemas==0.9.20260612
idna==3.18
kiwisolver==1.5.0
matplotlib==3.11.0
numpy==2.5.0
orjson==3.11.9
packaging==26.2
pillow==12.2.0
psutil==7.2.2
pybase64==1.4.3
pycparser==3.0
pydantic==2.13.4
pydantic_core==2.46.4
PyJWT==2.13.0
pyparsing==3.3.2
pyspnego==0.12.1
python-dateutil==2.9.0.post0
qiskit==2.4.2
qiskit-aer==0.17.2
qiskit-ibm-runtime==0.47.0
requests==2.34.2
requests_ntlm==1.3.0
rustworkx==0.18.0
samplomatic==0.20.0
scipy==1.18.0
six==1.17.0
sspilib==0.5.0
stevedore==5.8.0
typing-inspection==0.4.2
typing_extensions==4.15.0
urllib3==2.7.0
```

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/codebykrejci/quantum_oavqe.git
cd quantum_oavqe
pip install -r requirements.txt
```

## Usage

The project is divided into two main phases: **calculation** and **visualisation**.

---

### 1. Configuration Setup (`config_CuO2.py`, `config_bg.py`, `config_Si.py`)

Before running the calculations, parameters must be set in the **‘config’ files**. There are three **‘config’ files**, one for each material:

- `config_CuO2.py` for the 2D square lattice with a three-atom copper–oxygen (CuO₂) basis  
- `config_bg.py` for bilayer graphene  
- `config_Si.py` for the 3D $\text{sp}^{3}\text{s}^{*}$ silicon model  

The **‘config’ files** contain the calculation parameters and classical optimiser settings, such as:

- Optimiser method  
- Maximum number of iterations  
- Number of quantum circuit executions  
- Bootstrapping  

Additionally, they include material-specific parameters, such as on-site energies, hopping amplitudes, and the high-symmetry path in the first Brillouin zone along which the calculation is performed.

The high-symmetry path is divided into a **‘classical’ path** and a **‘quantum’ path**:

- The classical path is densely sampled and treated as continuous; it is used to compute the band structure via exact diagonalisation.  
- The quantum path consists of only a small number of discrete k-points, reflecting the higher computational cost of the OAVQE calculation.



### 2. Running the calculation (`main.py`)

The **'main.py'** script serves as the centralised entry point for executing calculations. Instead of modifying parameters directly within the source code for different runs, you can configure the experiments using command-line arguments in the terminal. Every command requires two primary inputs:

  - The target material configuration passed as a positional argument (e.g., config_Si for Silicon, config_CuO2 for Copper Oxide and config_bg for bilayer graphene).

  - The specific routine to execute, designated by the --exp flag.

The script supports three primary experiment routines:

### 2.1. Band Structure Calculation (--exp run_oa_vqe)

This is the standard simulation run. The script loads the material configuration, loops through every wave vector ($k$-point) along the defined path, and applies the OAVQE algorithm to compute the corresponding energy levels. This is the primary routine used to plot overall electronic band structures. The code performs the exact diagonalisation as well to calculate the reference energies. During execution, you can select which measurement backend or protocol to apply via the **--protocol flag**:

 - **new**: The Constant Measurement Protocol described in https://arxiv.org/pdf/2511.04389  (default). This option imports `cmp.py` and uses **AerSimulator** from **qiskit_aer** as a backend to simulate a quantum circuit.

 - **old**: The standard qubit-wise commuting (QWC) grouping protocol. This option imports `omp.py` and uses **AerSimulator** from **qiskit_aer** as a backend to simulate a quantum circuit.

 - **statevector**: An exact, noise-free statevector simulation that bypasses shot-sampling entirely. 

There are several additional options such as  **--max_iter** and **--shots** which denote the maximum number of iteration for optimiser and the number of quantum circuit executions, respectively.

Run the script:

```bash
python main.py config_CuO2 --exp run_oa_vqe --protocol new --max_iter 100 --shots 8192
```

### 2.2. Optimal Parameter Extraction (--exp run_eigenstates_angles)

This routine uses an ideal **statevector** backend to find the exact variational angles $\boldsymbol{\theta}$ for the ground and higher-order excited states across the $k$-path. To ensure the convergence to the correct global minima, the statevector backend is by default coupled with **L-BFGS-B** local optimiser with small tolerance, $\text{tol}= 10^{-9}$, critetion and bootstrapping method.
 It saves these optimised angles into an HDF5 database file (e.g., optimised_angles_CuO2.h5) so they can be reloaded for statistical benchmark later as described in the main article. 
Run the script:

```bash
python main.py config_CuO2 --exp run_eigenstates_angles
```

### 2.3. Statistical Benchmark Eigenstates (--exp run_benchmark_ksweep_eigenstates)
This routine sweeps across every state and $k$-point along the material path under a fixed shot allocation to evaluate how the constant measurement protocol **new** performs vs the original QWC grouping protocol **old**. This routine requires the precalculated exact parameter angles from **--run_eigenstates_angles** experiment. The code calculates the exact energies, and the energies using OAVQE with **new** and **old** method. The code evaluates the mean energies, standard deviations, variances, absolute errors and ratio of variances between the **new** and **old**.

Run the script:

```bash
python main.py config_CuO2 --exp run_benchmark_ksweep_eigenstates --shots 8192 --trials 100
```

There are additional options suchs as **--shots** that denotes the total number of shot budget that **new** protocol spilts among $3$ measurement bases independed on the size of the system and **old** method splits it among $\mathcal{O}(N)$ measurement bases, where $N$ is the size of the system. The parameter **--trials** denote the number of experiments to be run.

### 3. Real Hardware Run (oa_vqe_ibm.py)
While **main.py** is used for local simulations and benchmarks, this dedicated script connects directly to real quantum computers on the IBM Quantum Cloud architecture via the Qiskit Runtime Service. It targets a physical QPUs to run hardware-backed VQE optimisations. This version supports only Constant Measurement Protocol **new**. It calculates all eigenstates for single wave vector $\boldsymbol{k}$. 

Run the script:

```bash
python oa_vqe_ibm.py 
```

Users must change the paramteres such as the material **config_CuO2**, **config_bg**, **config_Si**, optimiser settings or specific wave vector $\boldsymbol{k}$ inside the script. 

**Important:** Running this script requires an active IBM Quantum account with access to the specified backend and session mode. Ensure your credentials are authenticated via QiskitRuntimeService(name="your_profile_name") before launching. Beware that the calculation is time expensive and can take up to 60 or 150 minutes of QPU time for **CuO2** and **bg**, respectively. 

### 4. Visualising the results (`printer.py`)

The visualisation script allows plotting the stored results from **--run_oa_vqe** and comparing them with the reference (classical) values.
Upon execution, a file dialog will open to select the HDF5 results file.

Run the script:

```bash
python printer.py
```

This script automatically generates four plots:

- **plot_eigenvalues** — All calculated VQD energies (coloured circles) vs. exact energies (gray line).
- **plot_each_eigenvalues** — Calculated energies, with each energy state plotted in a different color.
- **plot_n_fun** — The number of function evaluations per energy state.
- **plot_calc_time** — A heatmap of the minimization time (in seconds) for each state and k-point.

Users may further modify or extend the generated plots and other actions directly within the `printer.py` script.

## 5. Examples

The files `results_20260528-164958.h5` and `results_20260528-181330.h5` contain example results from **--exp run_oa_vqe** for the copper–oxygen (CuO₂) system and bilayer graphene, respectively.  

The results can be easily opened and visualised by running the `printer.py` script. The calculations were performed with $N_{\text{max}} = 50$ and $N_{\text{shots}} = 8192$. Additionally, there are raw data used in the main article such as `optimised_angles_cuo2.h5`, `optimised_angles_bg.h5`, and `optimised_angles_si.h5` obtained from **--exp run_eigenstate_angles**. Lastly there are raw data from **--exp run_benchmark_ksweep_eigenstates** stored in several **statistical_benchmark_ksweep_eigenstates_material_Nshots.txt** files, two for each material, where the benchmarks were perfromed with $N_{\text{shots}} = 10^3, 10^4$, respectively.


## License
This project is released under the MIT License.

## Citation

If you use this code in your research or project, please cite it as:

```bibtex
@misc{Krejci2025Minimum,
  title        = {Minimum measurements quantum protocol for band structure calculation},
  author       = {Michal Krejčí and Lucie Krejčí and Ijaz Ahamed Mohammad and Martin Plesch and Martin Friák},
  year         = {2025},
  eprint       = {2511.04389},
  archivePrefix= {arXiv},
  primaryClass = {quant-ph},
  url          = {https://arxiv.org/abs/2511.04389},
  doi          = {10.48550/arXiv.2511.04389}
}
```


KREJČÍ, Michal, 2025. Quantum band structure calculation using OAVQE and Constant Measurement Protocol [online]. GitHub repository, https://github.com/codebykrejci/quantum_oavqe.git

