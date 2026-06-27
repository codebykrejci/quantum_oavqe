import time
import datetime
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp 
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import config_bg as config # or use config_CuO2

def cost_func_oa(
    parameters: np.ndarray | list[float], 
    estimator: Estimator, 
    transpiled_ansatz: QuantumCircuit, 
    hamiltonian: np.ndarray, 
    offdiag_terms: list[tuple[int, int]], 
    num_qubits: int, 
    threshold: float = 1e-3
) -> float:
    """
    Evaluates the energy eigenvalues using the constant measurement protocol OAVQE framework 
    by communicating with the deployed Qiskit Runtime Service backend.
    """

    def _make_isa_op(sparse_list: list[tuple[str, list[int], float]]) -> SparsePauliOp:
        op = SparsePauliOp.from_sparse_list(sparse_list, num_qubits)
        if transpiled_ansatz.layout is not None:
            op = op.apply_layout(transpiled_ansatz.layout, num_qubits=transpiled_ansatz.num_qubits)
        return op

    Z_ops = [_make_isa_op([("Z", [j], 1.0)]) for j in range(num_qubits)]
    
    job_z = estimator.run([(transpiled_ansatz, Z_ops, parameters)])
    evs_z = job_z.result()[0].data.evs
    probabilities = (1 - evs_z) / 2
    
    cost = 0.0
    for j in range(num_qubits):
        epsilon = hamiltonian[j, j].real
        if abs(epsilon) > 1e-5:
            cost += epsilon * probabilities[j]
            
    S = np.where(probabilities > threshold)[0]
    m = len(S)
    
    if m < 2:
        return float(cost)
        
    if not offdiag_terms:
        return float(cost)
        
    obs_XX = []
    obs_XY = []
    mxx_pairs = []
    mxy_pairs = []
    
    for u in range(m):
        for v in range(u + 1, m):
            if (u % 2) != (v % 2): 
                j_phys = S[u]
                l_phys = S[v]
                
                obs_XX.append(_make_isa_op([("XX", [j_phys, l_phys], 1.0)]))
                mxx_pairs.append((j_phys, l_phys))
                
                if (u % 2) == 0:
                    obs_XY.append(_make_isa_op([("XY", [j_phys, l_phys], 1.0)]))
                else:
                    obs_XY.append(_make_isa_op([("YX", [j_phys, l_phys], 1.0)]))
                mxy_pairs.append((j_phys, l_phys))
                
    evs_XX = []
    evs_XY = []
    if obs_XX and obs_XY:
        job_cross = estimator.run([
            (transpiled_ansatz, obs_XX, parameters),
            (transpiled_ansatz, obs_XY, parameters)
        ])
        res_cross = job_cross.result()
        evs_XX = res_cross[0].data.evs
        evs_XY = res_cross[1].data.evs
        
    dict_XX = {pair: val for pair, val in zip(mxx_pairs, evs_XX)}
    dict_XY = {}
    for idx, (j_phys, l_phys) in enumerate(mxy_pairs):
        u = np.where(S == j_phys)[0][0]
        dict_XY[(j_phys, l_phys)] = evs_XY[idx] if (u % 2 == 0) else -1.0 * evs_XY[idx]
        
    C_jl = {}
    for u in range(m):
        for v in range(u + 1, m):
            if (u % 2) != (v % 2): 
                j_phys = S[u]
                l_phys = S[v]
                C_jl[(j_phys, l_phys)] = dict_XX[(j_phys, l_phys)] + 1j * dict_XY[(j_phys, l_phys)]
                C_jl[(l_phys, j_phys)] = np.conj(C_jl[(j_phys, l_phys)])
                
    for u in range(m):
        for v in range(u + 1, m):
            if (u % 2) == (v % 2): 
                j_phys = S[u]
                l_phys = S[v]
                valid_ks = [S[w] for w in range(m) if (w % 2) != (u % 2)]
                if valid_ks:
                    k_phys = max(valid_ks, key=lambda k: probabilities[k])
                    prob_k = probabilities[k_phys]

                    numerator = C_jl[(j_phys, k_phys)] * C_jl[(k_phys, l_phys)]
                    denominator = 2 * prob_k
                    C_jl[(j_phys, l_phys)] = numerator / denominator 
                    C_jl[(l_phys, j_phys)] = np.conj(C_jl[(j_phys, l_phys)])
                    
    for (j, l) in offdiag_terms:
        if j in S and l in S:
            c_val = C_jl.get((j, l), 0.0j)
            cost += hamiltonian[j, l].real * c_val.real + hamiltonian[j, l].imag * c_val.imag
            
    return float(cost)


def _Agate(circuit: QuantumCircuit, qubit1: int, qubit2: int, param1: ParameterExpression, param2: ParameterExpression) -> None:
    gate = XXPlusYYGate(param1, param2)
    circuit.append(gate, [qubit1, qubit2])

def _build_ansatz_circuit(n: int, num_qubits: int, prev_params_list: list[np.ndarray]) -> tuple[QuantumCircuit, ParameterVector]:
    circuit = QuantumCircuit(num_qubits)
    circuit.x(n) 
    
    params = ParameterVector('theta', 2 * (num_qubits - 1 - n))
    
    p_idx = 0
    for i in range(n, num_qubits - 1):
        _Agate(circuit, i, i+1, params[p_idx], params[p_idx+1])
        p_idx += 2
        
    for prev_idx in range(n - 1, -1, -1):
        p_fixed = prev_params_list[prev_idx]
        ptr = 0
        for i in range(prev_idx, num_qubits - 1):
            _Agate(circuit, i, i+1, p_fixed[ptr], p_fixed[ptr+1])
            ptr += 2
            
    return circuit, params

def run_single_kpoint_test() -> None:
    num_qubits: int = config.NUM_QUBITS
    num_states: int = config.num_states

    k_point = config.path_q[25] # choose here the target wave vector k
    print("\n" + "="*80)
    print(f"       IBM CLOUD HARDWARE JOB DEPLOYMENT (MOMENTUM SPACE k={k_point})")
    print("\n" + "="*80)
    
    class_ham, exact_evals = config.hamiltonian(k_point, *config.PARAMS)
    
    offdiag_terms = []
    for j in range(num_qubits):
        for l in range(j + 1, num_qubits):
            if abs(class_ham[j, l]) > 1e-9:
                offdiag_terms.append((j, l))

    if not offdiag_terms:
        print("Optimisation Active: Off-diagonal terms are zero. Cross-terms skipped.")

    print("Connecting to IBM Quantum Cloud Architecture...")
    service = QiskitRuntimeService(name="Lucinka")
    backend = service.backend("ibm_kingston")
    print(f"Selected System Target Backend: {backend.name}")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    n_shots = getattr(config, 'NSHOTS', 8192)
    
    quantum_evals = []
    opt_parameters = []
    nfev_list = []  
    
    print(f"Number of shots: {n_shots} default sample shots.")
    print("\nStarting VQE Optimisation Session...")
    print("-" * 50)
    
    with Session(backend=backend, max_time="3h") as session:
        print(f"IBM Session opened successfully with ID: {session.session_id}")
        
        try:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = n_shots
            estimator.options.resilience_level = 1
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"
            estimator.options.twirling.enable_gates = True
            
            for n in range(num_states):
                print(f"\n--- Optimising State {n} ---")
                start_time = time.time()
                
                logical_ansatz, param_vec = _build_ansatz_circuit(n, num_qubits, opt_parameters)
                num_var_params = len(param_vec)
                transpiled_ansatz = pm.run(logical_ansatz)
                
                if num_var_params > 0:
                    initial_params = np.random.uniform(0, 2 * np.pi, num_var_params)
                    
                    result = minimize(
                        cost_func_oa,
                        x0=initial_params,
                        args=(estimator, transpiled_ansatz, class_ham, offdiag_terms, num_qubits),
                        method=config.calc_method,
                        options={'maxiter': config.max_iter, "initial_tr_radius": 0.6, "final_tr_radius": 1e-3},
                        tol=2e-2
                    )
                    
                    quantum_evals.append(result.fun)
                    opt_parameters.append(result.x)
                    nfev_list.append(result.nfev) 
                else:
                    val = cost_func_oa([], estimator, transpiled_ansatz, class_ham, offdiag_terms, num_qubits)
                    quantum_evals.append(val)
                    opt_parameters.append(np.array([]))
                    nfev_list.append(1)
                    
                print(f"State {n} completed in {time.time() - start_time:.2f} s")

            if len(quantum_evals) == num_states:
                print("\n=== Final Hardware Results Comparison ===")
                print(f"{'State':<7} | {'Exact Energy':<15} | {'Quantum Energy':<15} | {'Diff (Abs Error)':<17} | {'Evaluations (nfun)':<18}")
                print("-" * 85)
                
                exact_evals_sorted = np.sort(exact_evals)
                
                quantum_with_indices = [(q_val, idx) for idx, q_val in enumerate(quantum_evals)]
                quantum_with_indices.sort(key=lambda x: x[0]) 
                
                for i, (quant_e, orig_idx) in enumerate(quantum_with_indices):
                    exact_e = exact_evals_sorted[i]
                    error = abs(exact_e - quant_e)
                    nfev_val = nfev_list[orig_idx]
                    
                    print(f"{i:<7} | {exact_e:<15.6f} | {quant_e:<15.6f} | {error:<17.6e} | {nfev_val:<18}")
                    
                print("\n=== Optimal Variational Angles Found ===")
                print("-" * 50)
                for i in range(num_states):
                    angles = opt_parameters[i]
                    if len(angles) > 0:
                        formatted_angles = ", ".join([f"{angle:.5f}" for angle in angles])
                        print(f"State {i} Optimal Angles (\u03b8):\n  [{formatted_angles}]")
                    else:
                        print(f"State {i} Optimal Angles (\u03b8):\n  [No parameters optimised - static calculation configuration]")
            else:
                print("\n[INFO] Optimisation sequence completed partially. Structural matrix shapes mismatch.")

        except Exception as run_error:
            print(f"\n[CRITICAL RUN ERROR] Loop optimisation aborted due to error: {run_error}")
            
    print("\nIBM Quantum Session closed.")

if __name__ == '__main__':
    run_single_kpoint_test()