import numpy as np

# Optimizer settings: 
calc_method = 'COBYQA'              # type of solver for scipy.optimize.minimize. Note that changing the optimizer
                                    # to some different type would require change of options in vqd_sampler, namely: 
                                    #'initial_tr_radius', 'final_tr_radius' are specific to COBYQA and are
                                    # in general not supported in diffrent implementations of local optmizers.
                                
max_iter = int(50)                  # the maximum number of iterations to perform in scipy.optimize.minimize
NSHOTS = int(8192)                   # number of quantum circuit executions to sample the measurements
bootstrapping = True                # True for bootstrapping or False for calculations without bootstrapping

# Calculation setting: 
n_q = 2                            # the number of k-points in each high symmetry k-path for quantum path 
n_c = 500                           # the number of interpolated values between individual k-points in a quantum path, which then form the path for classical computation
num_states = 3                      # number of eigenvalues / eigenstates for k-point
NUM_QUBITS = 3                      # number of qubit is the same as the dimension of the hamiltonian matrix

# Material parameters and helper functions: 
LATTICE_CONSTANT = 1            
NEIGHBORS = LATTICE_CONSTANT  * np.array([
    [1, 0],
    [0, 1]
    ])                              # nearest neighbors to atom at (0, 0). CuO2 is a 2D square lattice with three atom basis.

POINTS = {
    'G':  np.array([0, 0]),
    'M': np.pi / LATTICE_CONSTANT * np.array([1, 1]),
    'X': np.pi / LATTICE_CONSTANT * np.array([1, 0]),
}                                   # high-symmetry points in the 1st Brillouin zone.

PARAMS = (1.3, 3.6 , 0.0)             # tpd, ed, ep. These are hopping amplitude and on-site energies taken from Fulde.

def hamiltonian(k, tpd, ed, ep):    # CuO2 hamiltonian matrix
    h12 = tpd*(1 - np.exp(-1j * k @ NEIGHBORS[0]))
    h13 = -tpd*(1 - np.exp(-1j * k @ NEIGHBORS[1]))
    H = np.array([[ed, h12, h13],
                  [np.conjugate(h12), ep, 0],
                  [np.conjugate(h13), 0, ep]])
    
    eigvals = np.linalg.eigvalsh(H)
    eigvals.sort()
    return H, eigvals

# linear path through reciprocal space
def linpath(a, b, n=50, endpoint=True):
    spacings = [np.linspace(start, end, num=n, endpoint=endpoint) for start, end in zip(a, b)]
    return np.stack(spacings, axis=-1)

def interpolate_path(n_c, calc_path, calc_plot, exact, plot):

    for i in range(len(calc_path)-1):
        #print(i, calc_plot[i], calc_plot[i + 1])
        segment = np.linspace(calc_path[i], calc_path[i + 1], n_c, endpoint=False)
        exact.extend(segment)
        steps =  np.linspace(calc_plot[i], calc_plot[i + 1], n_c, endpoint=False)
        plot.extend(steps)

    path_exact.append(calc_path[-1])
    path_exact_plot.append(calc_plot[-1])
    return exact, plot

# path for quantum computation
GM_q = linpath(POINTS['G'], POINTS['M'], n_q, endpoint=False)   # Gamma - M
MX_q = linpath(POINTS['M'], POINTS['X'], n_q, endpoint=False)   # M - X
XG_q = linpath(POINTS['X'], POINTS['G'], n_q +1, endpoint=True)   # X - Gamma
labels_name = ['Γ', 'M', 'X','Γ']
labels_position = [0, n_q , 2*n_q, 3*n_q]
path_q = np.vstack([GM_q, MX_q, XG_q])
path_q_plot = np.arange(0, len(path_q), 1)

# path for classical diagonalization
path_q0 = np.vstack([GM_q, MX_q, XG_q])
path_q0_plot = np.arange(0, len(path_q0), 1)
path_exact = []
path_exact_plot = []
path_exact, path_exact_plot = interpolate_path(n_c, path_q0, path_q0_plot, path_exact, path_exact_plot)
path_exact_plot = np.stack(path_exact_plot)
path_exact = np.vstack(path_exact)