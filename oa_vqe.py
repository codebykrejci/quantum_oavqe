from qiskit.circuit import QuantumCircuit, ParameterVector, ParameterExpression
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Statevector
import numpy as np
import omp, cmp                           


class OrthogonalAnsatzVQE:
    def __init__(
        self, 
        config, 
        method: str = 'new'
    ) -> None:
        """
        Initialises the OA-VQE framework.
        
        :param config: The material configuration profile containing hamiltonian, geometry and optimisation settings
        :param method: The cost function algorithm to evaluate energy. Options: 'statevector', 'old', 'new'. 
        """
        self.config = config
        self.num_qubits = config.NUM_QUBITS
        self.Nshots = config.NSHOTS          
        
        self.method_name = method.lower()
        method_map = {
            'statevector': self.cost_func_oa_statevector,
            'old': self.cost_func_oa_old,
            'new': self.cost_func_oa_new
        }
        
        if self.method_name not in method_map:
            raise ValueError(f"Unknown method '{method}'. Choose from: 'statevector', 'old', 'new'")
            
        self.cost_func = method_map[self.method_name]
   
    def _Agate(
        self, 
        circuit: QuantumCircuit, 
        qubit1: int, 
        qubit2: int, 
        param1: float | ParameterExpression, 
        param2: float | ParameterExpression
    ) -> None:
        """
        Applies an XXPlusYYGate to the specified qubits in the circuit.

        :param circuit: The quantum circuit being constructed.
        :param qubit1: Index of the first target qubit.
        :param qubit2: Index of the second target qubit.
        :param param1: The first parameter (theta) for the XXPlusYYGate.
        :param param2: The second parameter (beta) for the XXPlusYYGate.
        """
        gate = XXPlusYYGate(param1, param2)
        circuit.append(gate, [qubit1, qubit2])

    def _build_ansatz_circuit(
        self, 
        n: int, 
        prev_params_list: list[np.ndarray], 
        current_params: np.ndarray | list[float] | None = None
    ) -> tuple[QuantumCircuit, ParameterVector | None]:
        """
        Creates a circuit for the n-th excited state where n = 0, 1, 2,..., num_qubits - 1
        
        If current_params is None, layer n uses a symbolic ParameterVector (For VQE Loops).
        If current_params is a list/array of numbers, layer n is hardcoded with those numbers 
        (For Fixed One-Off Evaluations).

        :param n: The target state index (dictates the excitation qubit).
        :param prev_params_list: List of parameter arrays for fixed layers [0 to n-1].
        :param current_params: Numerical array for the current layer (None for variational mode).
        :returns: A tuple containing the constructed quantum circuit and the symbolic parameters 
                  (if in variational mode) or None (if in fixed mode).
        """
        circuit = QuantumCircuit(self.num_qubits)
        circuit.x(n) 
        
        if current_params is None:
            params_to_bind = ParameterVector('theta', 2 * (self.num_qubits - 1 - n))
            params = params_to_bind
        else:
            params_to_bind = current_params
            params = None 
            
        p_idx = 0
        for i in range(n, self.num_qubits - 1):
            self._Agate(circuit, i, i+1, params_to_bind[p_idx], params_to_bind[p_idx+1])
            p_idx += 2
            
        for prev_idx in range(n - 1, -1, -1):
            p_fixed = prev_params_list[prev_idx]
            ptr = 0
            for i in range(prev_idx, self.num_qubits - 1):
                self._Agate(circuit, i, i+1, p_fixed[ptr], p_fixed[ptr+1])
                ptr += 2
                
        return circuit, params
    
    def _generate_angles(
        self, 
        n: int, 
        active_qubits_count: int, 
        beta_val: float = 0.0
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Generates the numerical arrays for the analytical uniform/localised state profiles.

        :param n: The target state index.
        :param active_qubits_count: The number of active qubits involved in the layer.
        :param beta_val: The constant beta angle to use for the second parameter of the gates.
        :returns: A tuple containing a list of parameter arrays for fixed historical layers [0 to n-1] 
                  and a flat 1D numpy array representing the target parameters for layer n.
        """
        list_of_params = []
        
        for m in range(n + 1):
            num_angles_for_m = 2 * (self.num_qubits - 1 - m)
            controlled_angles = np.zeros(num_angles_for_m)
            active_steps = min(active_qubits_count, self.num_qubits - m)
            
            for step in range(active_steps - 1):
                M_remaining = active_steps - step
                theta = 2 * np.arccos(1.0 / np.sqrt(M_remaining))
                
                p_idx = 2 * step
                if p_idx < num_angles_for_m:
                    controlled_angles[p_idx] = theta      
                    controlled_angles[p_idx + 1] = beta_val  
                    
            list_of_params.append(controlled_angles)

        prev_params_list = list_of_params[:n]
        active_thetas = list_of_params[n]
        
        return prev_params_list, active_thetas
    
    def _offdiag_terms(
        self, 
        hamiltonian: np.ndarray
    ) -> list[tuple[int, int]]:
        """
        Extracts the indices of non-zero off-diagonal terms from the given Hamiltonian.

        :param hamiltonian: The classical Hamiltonian matrix.
        :returns: A list of tuples, where each tuple contains the (j, l) indices of a non-zero off-diagonal term.
        """
        offdiag_terms = []
        for j in range(self.num_qubits):
            for l in range(j + 1, self.num_qubits):
                if abs(hamiltonian[j, l]) > 1e-5:
                    offdiag_terms.append((j, l))
        return offdiag_terms
    

    def cost_func_oa_statevector(
        self, 
        thetas: np.ndarray, 
        ansatz: QuantumCircuit, 
        hamiltonian: np.ndarray, 
        offdiag_terms: list[tuple[int, int]]
    ) -> float:
        """
        Evaluates the energy cost function using an exact statevector simulation.

        :param thetas: A 1D array of variational parameters.
        :param ansatz: The parametrised quantum circuit ansatz.
        :param hamiltonian: The Hamiltonian matrix representing the observable.
        :param offdiag_terms: List of qubit pairs corresponding to off-diagonal elements in the Hamiltonian.
        :returns: The computed energy expectation value.
        """
        param_circuit = ansatz.assign_parameters(thetas)
        state_vector = Statevector.from_instruction(param_circuit).data
        amplitudes = {
            j: state_vector[int(''.join(['0'] * (self.num_qubits - 1 - j) + ['1'] + ['0'] * j), 2)]
            for j in range(self.num_qubits)
        }
        
        diag_contr = sum(
            hamiltonian[j, j].real * abs(amplitudes[j])**2
            for j in range(self.num_qubits)
            if hamiltonian[j, j].real != 0
        )
        
        offdiag_contr = sum(
            2 * np.real(hamiltonian[j, l] * np.conjugate(amplitudes[j]) * amplitudes[l])
            for (j, l) in offdiag_terms
        )
        
        return diag_contr + offdiag_contr
    
    def cost_func_oa_old(
        self, 
        thetas: np.ndarray, 
        ansatz: QuantumCircuit, 
        hamiltonian: np.ndarray, 
        offdiag_terms: list[tuple[int, int]]
    ) -> float:
        """
        Evaluates the energy cost function using the original O(num_qubits) measurement protocol

        :param thetas: A 1D array of variational parameters.
        :param ansatz: The parametrized quantum circuit ansatz.
        :param hamiltonian: The Hamiltonian matrix representing the observable.
        :param offdiag_terms: List of qubit pairs corresponding to off-diagonal elements in the Hamiltonian.
        :returns: The computed energy expectation value.
        """
        bound_circuit = ansatz.assign_parameters(thetas)
    
        probabilities = omp.amplitudes(bound_circuit, self.Nshots)
        expvals_xx = omp.expval_xx(bound_circuit, self.Nshots, offdiag_terms)
        expvals_xy = omp.expval_xy(bound_circuit, self.Nshots, offdiag_terms)

        cost = 0.0

        for j in range(self.num_qubits):
            epsilon = hamiltonian[j, j].real
            if abs(epsilon) > 1e-5:
                cost += epsilon * probabilities[j]
        
        for (j, l) in offdiag_terms:
            cost += hamiltonian[j, l].real * expvals_xx.get((j, l), 0.0)
            cost -= hamiltonian[j, l].imag * expvals_xy.get((j, l), 0.0)
            
        return cost
    
    def cost_func_oa_new(
        self, 
        thetas: np.ndarray, 
        ansatz: QuantumCircuit, 
        hamiltonian: np.ndarray, 
        offdiag_terms: list[tuple[int, int]]
    ) -> float:
        """
        Evaluates the energy cost function using the new O(1) constant measurement protocol.

        :param thetas: A 1D array of variational parameters.
        :param ansatz: The parametrized quantum circuit ansatz.
        :param hamiltonian: The Hamiltonian matrix representing the observable.
        :param offdiag_terms: List of qubit pairs corresponding to off-diagonal elements in the Hamiltonian.
        :returns: The computed energy expectation value.
        """
        bound_circuit = ansatz.assign_parameters(thetas)

        probabilities, amplitudes_abs, nonzero_keys, nonzero_pairs = cmp.amplitudes(bound_circuit, self.Nshots)
        expvals_xx = cmp.expectation_values_xx(bound_circuit, self.Nshots, nonzero_keys)
        expvals_xy = cmp.expectation_values_xy(bound_circuit, self.Nshots, nonzero_keys)
        complex_corr = cmp.C(expvals_xx, expvals_xy)

        cost = 0.0
    
        for j in range(self.num_qubits):
            epsilon = hamiltonian[j, j].real
            if abs(epsilon) > 1e-5:
                cost += epsilon * probabilities[j]
        
        if len(nonzero_keys) > 1:
            hamiltonian_active_pairs = set(offdiag_terms) 
         
            for (j, l) in nonzero_pairs:

                if ((j, l) not in hamiltonian_active_pairs or 
                    amplitudes_abs[j] < 1e-3 or 
                    amplitudes_abs[l] < 1e-3):
                    continue 
            
                if (j, l) in complex_corr:
                    cost += (hamiltonian[j, l] * complex_corr[(j, l)]).real
                
                else:
                    valid_k = [k_mid for k_mid in range(j + 1, l) 
                                if (j, k_mid) in complex_corr and (k_mid, l) in complex_corr]
                    if valid_k:
                        best_k = max(valid_k, key=lambda k: probabilities[k])
                        Cjl = complex_corr[(j, best_k)] * complex_corr[(best_k, l)] / (2 * probabilities[best_k])
                        cost += (hamiltonian[j, l] * Cjl).real
        return cost