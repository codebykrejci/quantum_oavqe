import numpy as np 

# Optimizer settings:

calc_method = 'COBYQA'              # type of solver for scipy.optimize.minimize 
max_iter = 1000                     # the maximum number of iterations to perform in scipy.optimize.minimize
NSHOTS = int(8192)                # number of quantum circuit executions to sample the measurements
bootstrapping = True                # True for bootstrapping or False for calculations without bootstrapping

# Calculation setting:
num_states = 10                     # the number of requested calculated states, max 10 with given Hamiltonian
n_q = 15                            # the number of k-points in each* high symmetry k-path for quantum path *except for X-K, where it is n_c//4 -> n_c should be >= 4
n_c = 1000                          # the number of interpolated values between individual k-points in a quantum path, which then form the path for classical computation
NUM_QUBITS = 10                     # number of qubits

# Material parameters and helper functions: 
LATTICE_CONSTANT = 5.43095  # in Angstroms 
NEIGHBORS = LATTICE_CONSTANT / 4 * np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1]
])                                  # nearest neighbors to atom at (0, 0, 0)

# on-site energies and hopping amplitudes for sp3s* Si tight-binding model taken from Vogl.
PARAMS = (-4.200, 1.7150, -4.200, 1.7150, 6.6850, 6.6850, -8.3000 , 1.7150, 4.5750, 5.7292, 5.7292, 5.3749, 5.3748)

# symmetry points at 1st Brillouin zone of the fcc lattice
POINTS = {
    'G': 2 * np.pi / LATTICE_CONSTANT * np.array([0, 0, 0]),
    'L': 2 * np.pi / LATTICE_CONSTANT * np.array([1/2, 1/2, 1/2]),
    'K': 2 * np.pi / LATTICE_CONSTANT * np.array([3/4, 3/4, 0]),
    'X': 2 * np.pi / LATTICE_CONSTANT * np.array([0, 0, 1]),
    'W': 2 * np.pi / LATTICE_CONSTANT * np.array([1, 1/2, 0]),
    'U': 2 * np.pi / LATTICE_CONSTANT * np.array([1/4, 1/4, 1]),
}
def phase(k, neighbors):  
        a, b, c, d = [np.exp(1j * k @ neighbor) for neighbor in neighbors]
        factors = np.array([
            a + b + c + d,
            a + b - c - d,
            a - b + c - d,
            a - b - c + d
        ])
        return (1/4) * factors
    
def hamiltonian(k, Esa, Epa, Esc, Epc, Essa, Essc, vss,  vxx, vxy, vsapc, vscpa, vssapc, vpassc):   
    g = phase(k, NEIGHBORS)
    gc = np.conjugate(g)
    H = np.array([
                   #sa            sc                 pax              pay              paz      pcx            pcy            pcz                  ssa                ssc
        [          Esa,    vss * g[0],               0,               0,                0,  vsapc * g[1],  vsapc * g[2],   vsapc * g[3],               0,              0],   # sa
        [  vss * gc[0],           Esc,  -vscpa * gc[1],  -vscpa * gc[2],   -vscpa * gc[3],             0,             0,              0,               0,              0],   # sc
        [            0, -vscpa * g[1],             Epa,               0,                0,    vxx * g[0],    vxy * g[3],     vxy * g[2],               0,  -vpassc * g[1]],  # pax
        [            0, -vscpa * g[2],               0,             Epa,                0,    vxy * g[3],    vxx * g[0],     vxy * g[1],               0,  -vpassc * g[2]],  # pay
        [            0, -vscpa * g[3],               0,               0,              Epa,    vxy * g[2],    vxy * g[1],     vxx * g[0],               0,  -vpassc * g[3]],  #paz
        [vsapc * gc[1],             0,     vxx * gc[0],     vxy * gc[3],      vxy * gc[2],           Epc,             0,              0,  vssapc * gc[1],               0],  #pcx
        [vsapc * gc[2],             0,     vxy * gc[3],     vxx * gc[0],      vxy * gc[1],             0,           Epc,              0,  vssapc * gc[2],               0],  #pcy
        [vsapc * gc[3],             0,     vxy * gc[2],     vxy * gc[1],      vxx * gc[0],             0,             0,            Epc,  vssapc * gc[3],               0],  #pcz
        [            0,             0,               0,               0,                0, vssapc * g[1], vssapc * g[2],  vssapc * g[3],            Essa,               0],  #ssa
        [            0,             0, -vpassc * gc[1], -vpassc * gc[2],  -vpassc * gc[3],             0,             0,              0,               0,            Essc]   #ssc
    ])

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
lambd_q = linpath(POINTS['L'], POINTS['G'], n_q, endpoint=False)        # λ - Γ
delta_q = linpath(POINTS['G'], POINTS['X'], n_q, endpoint=False)        # Γ - X
x_uk_q = linpath(POINTS['X'], POINTS['U'], n_q //4  , endpoint=False)   # X - U
sigma_q = linpath(POINTS['K'], POINTS['G'], n_q, endpoint=True)         # K - Γ
labels_name = ['L', 'Γ', 'X', 'U,K', 'Γ']
labels_position = [0, n_q , 2*n_q , 2*n_q + n_q//4 , 3*n_q + n_q//4 -1]
path_q = np.vstack([lambd_q, delta_q, x_uk_q, sigma_q])
path_q_plot = np.arange(0, len(path_q),1)

# path for classical diagonalization
path_q0 = np.vstack([lambd_q, delta_q, x_uk_q])
path_q0_plot = np.arange(0, len(path_q0), 1)
sigma_plot = np.arange(len(path_q0_plot),len(path_q_plot),1)
path_exact = []
path_exact_plot = []
path_exact, path_exact_plot = interpolate_path(n_c, path_q0, path_q0_plot, path_exact, path_exact_plot)
path_exact, path_exact_plot = interpolate_path(n_c, sigma_q, sigma_plot, path_exact, path_exact_plot)
path_exact_plot = np.stack(path_exact_plot)
path_exact = np.vstack(path_exact)