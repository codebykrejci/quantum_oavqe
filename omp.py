from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator
from collections import defaultdict

SIMULATOR = AerSimulator(method="statevector")

def M_z(
    quantum_circuit: QuantumCircuit, 
    Nshots: int
) -> dict[str, int]:
    """
    Performs a computational (Z) basis measurement on all qubits.

    Creates a copy of the provided quantum circuit, measures all qubits, 
    and executes it on the statevector simulator.

    :param quantum_circuit: The quantum circuit to be measured.
    :param Nshots: The number of measurement shots for the simulation.
    :returns: A dictionary mapping measured bitstrings to their occurrence counts.
    """
    qc = quantum_circuit.copy()
    qc.measure_all()
    result = SIMULATOR.run(qc, shots=Nshots).result()
    counts = result.get_counts()
    
    return counts
    

def M_xx(
    quantum_circuit: QuantumCircuit, 
    Nshots: int, 
    offdiag_terms: list[tuple[int, int]]
) -> dict[str, int]:
    """
    Performs an XX-basis measurement on the subsets of qubits specified in offdiag_terms.

    Extracts all unique qubits from the provided pairs, applies a Hadamard (H) gate 
    to each to rotate into the X-basis, and then measures them. If no terms are 
    provided, it returns an empty dictionary.

    :param quantum_circuit: The quantum circuit representing the state to be measured.
    :param Nshots: The number of measurement shots.
    :param offdiag_terms: A list of qubit pairs (tuples) defining which qubits to measure.
    :returns: Measurement counts for the target qubits.
    """
    target_qubits = sorted(list(set([q for pair in offdiag_terms for q in pair])))
    
    if not target_qubits:
        return {}

    qc = quantum_circuit.copy()
    
    creg = ClassicalRegister(len(target_qubits), name='meas')
    qc.add_register(creg)
    qc.barrier()

    for q in target_qubits:
        qc.h(q)

    for i, q in enumerate(target_qubits):
        qc.measure(q, creg[i])

    result = SIMULATOR.run(qc, shots=Nshots).result()
    counts = result.get_counts()
    
    return counts


def M_xy(
    quantum_circuit: QuantumCircuit, 
    Nshots: int, 
    offdiag_terms: list[tuple[int, int]]
) -> dict[str, dict[str, int]]:
    """
    Performs XY-basis measurements grouped by Qubitwise Commuting (QWC) sets.

    Groups the pairs in "offdiag_terms" by their first index (j). For each group,
    a Hadamard gate is applied to qubit "j" (X-basis), and a sqrt(X) (SX) gate 
    is applied to all associated "l" targets (Y-basis). A separate circuit is 
    executed for each QWC group.

    :param quantum_circuit: The quantum circuit representing the state to be measured.
    :param Nshots: The number of measurement shots per group.
    :param offdiag_terms: A list of qubit pairs (j, l) where j is measured in X and l in Y.
    :returns: A dictionary mapping group names (e.g., "group_0") to their respective measurement counts.
    """
    if not offdiag_terms:
        return {}

    qwc_groups = defaultdict(set)
    for j, l in offdiag_terms:
        qwc_groups[j].add(l)
    
    all_results = {}

    for j, targets in qwc_groups.items():
        qc = quantum_circuit.copy()
        
        unique_qubits = sorted(list(targets | {j}))
        creg = ClassicalRegister(len(unique_qubits), name=f'meas_j{j}')
        qc.add_register(creg)
        qc.barrier()
        
        idx_map = {q: i for i, q in enumerate(unique_qubits)}
        
        qc.h(j)
        for l in targets:
            # qc.sdg(l)
            # qc.h(l)
            qc.sx(l)
            
        for q in unique_qubits:
            qc.measure(q, creg[idx_map[q]])
            
        result = SIMULATOR.run(qc, shots=Nshots).result()
        all_results[f"group_{j}"] = result.get_counts()

    return all_results


def amplitudes(
    ansatz: QuantumCircuit, 
    Nshots: int
) -> dict[int, float]:
    """
    Estimates the probabilities |a_j|^2 for single-excitation basis states.

    Executes a Z-basis measurement on the circuit and filters the results for 
    computational basis states containing a single '1' (representing an excitation). 
    It calculates the probability of each single-excitation state based on shot counts.

    :param ansatz: Quantum circuit representing the variational state.
    :param Nshots: Number of measurement shots.
    :returns: A dictionary mapping the qubit index (where the excitation resides) to its probability.
    """
    counts = M_z(ansatz, Nshots)
    N = ansatz.num_qubits
    probabilities = {}

    for j in range(N):
        bitstring = ['0'] * N
        bitstring[j] = '1'
        bitstring = ''.join(bitstring)[::-1]
        count = counts.get(bitstring, 0)
        p = count / Nshots if count > 0 else 0.0
        probabilities[j] = p

    return probabilities


def expval_xx(
    ansatz: QuantumCircuit, 
    Nshots: int, 
    offdiag_terms: list[tuple[int, int]]
) -> dict[tuple[int, int], float]:
    """
    Computes the expectation values <X_j X_l> for given qubit pairs.

    :param ansatz: Quantum circuit representing the variational state.
    :param Nshots: Number of measurement shots.
    :param offdiag_terms: List of qubit pairs (tuples) for which to compute the expectation value.
    :returns: Dictionary mapping qubit pairs (j, l) to their evaluated <X_j X_l> expectation values.
    """
    target_qubits = sorted(list(set([q for pair in offdiag_terms for q in pair])))
    idx_map = {q: i for i, q in enumerate(target_qubits)}
    
    counts = M_xx(ansatz, Nshots, offdiag_terms)
    expectation_dict = {}

    for j, l in offdiag_terms:
        expval = 0.0
        for bitstring, count in counts.items():
            bits = [int(b) for b in bitstring[::-1]]
            parity = bits[idx_map[j]] ^ bits[idx_map[l]]   
            expval += ((-1) ** parity) * (count / Nshots)    
        expectation_dict[(j, l)] = expval

    return expectation_dict


def expval_xy(
    ansatz: QuantumCircuit, 
    Nshots: int, 
    offdiag_terms: list[tuple[int, int]]
) -> dict[tuple[int, int], float]:
    """
    Computes the expectation values <X_j Y_l> for given qubit pairs.

    Utilizes the group-wise measurements generated by "M_xy" to reconstruct 
    the expectation values. 

    :param ansatz: Quantum circuit representing the variational state.
    :param Nshots: Number of measurement shots per QWC group.
    :param offdiag_terms: List of qubit pairs (tuples) for which to compute the <X_j Y_l> expectation value.
    :returns: Dictionary mapping qubit pairs (j, l) to their evaluated <X_j Y_l> expectation values.
    """
    all_counts = M_xy(ansatz, Nshots, offdiag_terms)
    expectation_dict = {}

    qwc_groups = defaultdict(set)
    for j, l in offdiag_terms:
        qwc_groups[j].add(l)

    for j, targets in qwc_groups.items():
        unique_qubits = sorted(list(targets | {j}))
        idx_map = {q: i for i, q in enumerate(unique_qubits)}       
        counts = all_counts[f"group_{j}"]
        
        for l in targets:
            expval = 0.0
            for bitstring, count in counts.items():
                bits = [int(b) for b in bitstring[::-1]]
                parity = bits[idx_map[j]] ^ bits[idx_map[l]]
                expval += ((-1) ** parity) * (count / Nshots)
            
            expectation_dict[(j, l)] = expval

    return expectation_dict