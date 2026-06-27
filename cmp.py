from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
import itertools

SIMULATOR = AerSimulator(method="statevector")

def M_z(
    quantum_circuit: QuantumCircuit, 
    Nshots: int
) -> dict[str, int]:      
    """
    Performs a measurement of all qubits in the computational (Z) basis.
    The quantum circuit is copied, all qubits are measured, and the circuit
    is executed on a statevector-based Aer simulator for a given number of shots.

    :param quantum_circuit: Quantum circuit to be measured.
    :param Nshots: Number of measurement shots used to estimate probabilities.
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
    nonzero_keys: list[int]
) -> dict[str, int]: 
    """
    Performs an XX-basis measurement on a selected subset of qubits based on nonzero_keys.
    Hadamard gates are applied to the qubits specified by "nonzero_keys",
    followed by measurement in the computational basis. 
    If fewer than two nonzero_keys are provided, an empty dictionary is returned.

    :param quantum_circuit: Quantum circuit representing the state to be measured.
    :param Nshots: Number of measurement shots.
    :param nonzero_keys: Indices of qubits to be measured in the X basis.
    :returns: Measurement counts for the selected qubits.
    """
    if len(nonzero_keys) <= 1:
        return {}
    
    else:    
        qc = quantum_circuit.copy()
        creg = ClassicalRegister(len(nonzero_keys))
        qc.add_register(creg)
        qc.barrier()

        for q in nonzero_keys:
            qc.h(q)

        for i, q in enumerate(nonzero_keys):
            qc.measure(q, creg[i])

    result = SIMULATOR.run(qc, shots=Nshots).result()
    counts = result.get_counts()

    return counts

def M_xy(
    quantum_circuit: QuantumCircuit, 
    Nshots: int, 
    nonzero_keys: list[int]
) -> dict[str, int]: 
    """
    Performs an XY-basis measurement on a selected subset of qubits.
    For qubits at even positions in "nonzero_keys", a Hadamard gate is applied
    (X measurement). For qubits at odd positions, sqrt(X) (SX) gate
    is applied (Y measurement). The qubits are then measured in
    the computational basis.
    If fewer than two "nonzero_keys" are provided, an empty dictionary is returned.

    :param quantum_circuit: Quantum circuit representing the state to be measured.
    :param Nshots: Number of measurement shots.
    :param nonzero_keys: Indices of qubits to be measured in alternating X/Y bases.
    :returns: Measurement counts for the selected qubits.
    """
    if len(nonzero_keys) <= 1:
        return {}
    
    else:
        qc = quantum_circuit.copy()
        creg = ClassicalRegister(len(nonzero_keys))
        qc.add_register(creg)
        qc.barrier()

        for i, q in enumerate(nonzero_keys):
            if i % 2 == 0:
                qc.h(q)
            else:
                #qc.sdg(q)
                #qc.h(q)
                qc.sx(q)
                
        for i, q in enumerate(nonzero_keys):
            qc.measure(q, creg[i])

    result = SIMULATOR.run(qc, shots=Nshots).result()
    counts = result.get_counts()

    return counts

def amplitudes(
    ansatz: QuantumCircuit, 
    Nshots: int
) -> tuple[dict[int, float], dict[int, float], list[int], list[tuple[int, int]]]:
    """
    Estimates probabilities |a_{j}|^2 and amplitudes |a_{j}| from Z-basis measurements.
    This function extracts probabilities and absolute values of amplitudes for single-excitation basis states.
    It also identifies qubits with nonzero amplitudes and all possible pairs among them.

    :param ansatz: Quantum circuit representing the variational state.
    :param Nshots: Number of measurement shots.
    :returns: A tuple containing:
        - probabilities: Probability of finding the excitation on each qubit.
        - amplitudes_abs: Absolute value of the corresponding amplitudes.
        - nonzero_keys: Indices of qubits with non-negligible amplitudes.
        - nonzero_pairs: All unique pairs of indices from ``nonzero_keys``.
    """
    eps_zero = 1e-3 
    counts = M_z(ansatz, Nshots)
    N = ansatz.num_qubits
    probabilities = {}
    amplitudes_abs = {}

    for j in range(N):
        bitstring = ['0'] * N
        bitstring[j] = '1'
        bitstring = ''.join(bitstring)[::-1]
        count = counts.get(bitstring, 0)
        p = count / Nshots if count > 0 else 0.0
        probabilities[j] = p
        amplitudes_abs[j] = np.sqrt(p) if p > 0 else 0.0

    nonzero_keys = [k for k, v in amplitudes_abs.items() if v >= eps_zero]
    nonzero_pairs = list(itertools.combinations(nonzero_keys, 2))

    return probabilities, amplitudes_abs, nonzero_keys, nonzero_pairs

def expectation_values_xx(
    ansatz: QuantumCircuit, 
    Nshots: int, 
    nonzero_keys: list[int]
) -> dict[tuple[int, int], float]:
    """
    Computes expectation values ⟨X_i X_j⟩ for selected qubit pairs.
    The expectation values are estimated from measurement statistics obtained
    by performing XX-basis measurements on the qubits listed in ``nonzero_keys``.

    :param ansatz: Quantum circuit representing the variational state.
    :param Nshots: Number of measurement shots.
    :param nonzero_keys: Indices of qubits included in the measurement.
    :returns: Dictionary mapping qubit index pairs (i, j) to ⟨X_i X_j⟩.
    """
    counts = M_xx(ansatz, Nshots, nonzero_keys)
    expectation_dict = {}

    for p in range(len(nonzero_keys) - 1):
        for q in range(p + 1, len(nonzero_keys)):
            j = nonzero_keys[p]
            l = nonzero_keys[q]
            expval = 0.0

            for bitstring, count in counts.items():
                bits = [int(b) for b in bitstring[::-1]]
                parity = bits[p] ^ bits[q]
                expval += ((-1) ** parity) * (count / Nshots)
            expectation_dict[(j, l)] = expval

    return expectation_dict

def expectation_values_xy(
    ansatz: QuantumCircuit, 
    Nshots: int, 
    nonzero_keys: list[int]
) -> dict[tuple[int, int], float | None]:
    """
    Computes expectation values ⟨X_i Y_j⟩ for selected qubit pairs.
    The expectation values are reconstructed from XY-basis measurements,
    accounting for the ordering of qubits and the sign convention arising
    from measuring ⟨Y_i X_j⟩ in certain configurations.

    :param ansatz: Quantum circuit representing the variational state.
    :param Nshots: Number of measurement shots.
    :param nonzero_keys: Indices of qubits included in the measurement.
    :returns: Dictionary mapping qubit index pairs (i, j) to ⟨X_i Y_j⟩.
              Entries may be ``None`` when the measurement configuration
              does not yield the desired correlator.
    """
    counts = M_xy(ansatz, Nshots, nonzero_keys)
    expectation_dict = {}

    for p in range(len(nonzero_keys) - 1):
        for q in range(p + 1, len(nonzero_keys)):
            j = nonzero_keys[p]
            l = nonzero_keys[q]
            expval = 0.0

            for bitstring, count in counts.items():
                bits = [int(b) for b in bitstring[::-1]]
                parity = bits[p] ^ bits[q]
                expval += ((-1) ** parity) * (count / Nshots)

            if (p % 2 == 0) and (q % 2 == 1):
                xy = expval

            elif (p % 2 == 1) and (q % 2 == 0):
                xy = -expval

            else:
                xy = None
            expectation_dict[(j, l)] = xy

    return expectation_dict

def C(
    exp_xx: dict[tuple[int, int], float], 
    exp_xy: dict[tuple[int, int], float | None]
) -> dict[tuple[int, int], complex]:     
    """
    Constructs the complex correlator C_{j,l} = ⟨X_j X_l⟩ + i⟨X_j Y_l⟩.
    The complex correlator is built by combining the real XX and imaginary XY
    expectation values for each qubit pair.

    :param exp_xx: Dictionary of XX expectation values.
    :param exp_xy: Dictionary of XY expectation values.
    :returns: Dictionary mapping qubit index pairs to complex correlators.
    """
    C_dict = {}

    for (j, l), xx_val in exp_xx.items():
        xy_val = exp_xy.get((j, l), None)

        if xy_val is not None:
            C_dict[(j, l)] = xx_val + 1j * xy_val
            
    return C_dict