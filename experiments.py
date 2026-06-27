import datetime
import time
import h5py
import numpy as np
from scipy.optimize import minimize
from qiskit import transpile
import cmp

simulator = cmp.SIMULATOR


class OAVQEExperiments:
    def __init__(self, vqe_solver: 'OrthogonalAnsatzVQE', **kwargs) -> None:
        """
        Initialises the OAVQEExperiments framework to orchestrate VQE runs over parameter paths.
        
        :param vqe_solver: An initialised instance of the core OrthogonalAnsatzVQE solver.
        :param kwargs: Additional configuration overrides.
        """
        self.vqe_solver = vqe_solver
        self.config = vqe_solver.config  
        self.num_qubits = vqe_solver.num_qubits
        self.hpar = self.config.PARAMS
        self.num_states = self.config.num_states
        self.bootstrapping = getattr(self.config, 'bootstrapping', True)
        self.calc_method = getattr(self.config, 'calc_method', 'COBYQA')
        self.max_iter = getattr(self.config, 'max_iter', 1000)
        self.path_q = getattr(self.config, 'path_q', [])
        self.exact_path_q = getattr(self.config, 'path_exact', [])
        self.protocol = kwargs.get('protocol', getattr(self.config, 'protocol', 'new'))
        self.step = 0
        self.calc_optimal_param = {}

    def energies_q(
        self, 
        k: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float]]:
        """
        Calculates the quantum energies for a given wave vector k across multiple orthogonal states.

        Iteratively constructs the variational ansatz for each state, transpiles it, 
        and uses a classical optimiser to minimize the energy cost function. Supporting 
        bootstrapping initial parameters from previous steps if configured.

        :param k: The physical parameter (wave vector) for the Hamiltonian.
        :returns: A tuple containing:
            - eigenvalues: An array of optimised energy values for each state.
            - n_fun: An array of the number of cost function evaluations per state.
            - opt_parameters: A list of the optimised parameter arrays for each state.
            - time_duration: A list of execution times (in seconds) for each state's minimisation.
        """
        eigenvalues = []
        opt_parameters = []
        n_fun = []
        time_duration = []
        
        class_ham, _ = self.config.hamiltonian(k, *self.hpar)
        offdiag_terms = self.vqe_solver._offdiag_terms(class_ham)

        for n in range(self.num_states):
            print(f'Step {n + 1}: Starting minimisation with [{self.calc_method}] method using [{self.protocol}]')
            start_time_minimize = time.time()

            ansatz_template, param_vec = self.vqe_solver._build_ansatz_circuit(n, opt_parameters)
            transpiled_template = transpile(ansatz_template, simulator)
            num_var_params = len(param_vec) if param_vec is not None else 0

            # Determine initial parameter guess: bootstrapping from previous k-point or uniform random sampling
            if (self.step > 0) and (self.bootstrapping):
                initial_parameters = self.calc_optimal_param[n]
            else:
                initial_parameters = np.random.uniform(0, 2 * np.pi, num_var_params)

            minimize_args = (transpiled_template, class_ham, offdiag_terms)

            # If the layer has variational parameters, optimise it; otherwise, evaluate directly
            if num_var_params > 0:
                result = minimize(
                    self.vqe_solver.cost_func,
                    x0=initial_parameters,
                    args=minimize_args,
                    method=self.calc_method,    
                    options={'maxiter': self.max_iter, "initial_tr_radius": 0.5, "final_tr_radius": 1e-3},  # these are only specific COBYQA options
                    tol=1e-3 
                )
                
                eigenvalues.append(result.fun)
                opt_parameters.append(result.x)
                n_fun.append(result.nfev)
            else:
                val = self.vqe_solver.cost_func([], *minimize_args)
                eigenvalues.append(val)
                opt_parameters.append(np.array([]))
                n_fun.append(0)

            end_time_minimize = time.time()
            duration_minimize = end_time_minimize - start_time_minimize
            time_duration.append(duration_minimize)
            print(f'Minimisation done, duration: {duration_minimize:.4f} sec')

        return np.array(eigenvalues), np.array(n_fun), opt_parameters, time_duration

    def run_calculation(self) -> None:
        """
        Executes the full experiment sweep over the defined path of k-points.
        
        Computes exact classical baseline eigenvalues, runs the quantum VQE routine at each k-point, 
        and continuously streams and flushes all metadata, durations, and results into an HDF5 file.
        """
        start_time_script = time.time()
        self.step = 0
        timename = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        with h5py.File(f'results_{timename}.h5', 'w') as f:
            f.create_dataset('cost_method', data=self.vqe_solver.method_name, dtype=h5py.string_dtype(encoding='utf-8'))
            f.create_dataset('calc_method', data=self.calc_method, dtype=h5py.string_dtype(encoding='utf-8'))
            f.create_dataset('max_iter', data=self.max_iter)
            f.create_dataset('bootstrapping', data=self.bootstrapping)
            f.create_dataset('num_states', data=self.num_states)
            f.create_dataset('lattice_constant', data=self.config.LATTICE_CONSTANT)
            f.create_dataset('path_print', data=self.config.path_q_plot)
            f.create_dataset('labels_position', data=self.config.labels_position)
            f.create_dataset('labels_name', data=self.config.labels_name, dtype=h5py.string_dtype(encoding='utf-8'))

            print('Calculating exact values...')
            exact_values = []
            for k_ in self.exact_path_q:
                _, class_eignvls = self.config.hamiltonian(k_, *self.hpar)
                exact_values.append(class_eignvls)

            exact_group = f.create_group('exact_values')
            exact_group.create_dataset('eigenvalues', data=exact_values)
            exact_group.create_dataset('path', data=self.config.path_exact_plot)
            print('Done.')
            
            print(f'Starting quantum computation using [{self.protocol}] protocol...')
            start_calc_time = time.time()

            for k_ in range(len(self.path_q)):
                k_group = f.create_group(f'k-point index {k_}')
                print(f'Calculating {k_+1}/{len(self.path_q)}')
                result = self.energies_q(self.path_q[k_])

                k_group.create_dataset('eigenvalues', data=result[0])
                k_group.create_dataset('n_fun', data=result[1])
                
                dt = h5py.vlen_dtype(np.dtype('float64'))
                k_group.create_dataset('optimal_params', data=result[2], dtype=dt)
                k_group.create_dataset('minimize_time', data=result[3]) 

                f.flush() 

                # Store current optimal parameters for potential bootstrapping in the next k-point step
                self.calc_optimal_param = result[2]
                self.step += 1

            end_calc_time_script = time.time()
            duration_calc = end_calc_time_script - start_calc_time
            f.create_dataset('calculated_values_duration', data=duration_calc)

        print(f'Total execution duration: {time.time() - start_time_script:.4f} sec')

    def statistical_benchmark_kpoint(
        self, 
        k_point: float | np.ndarray, 
        state_index: int = 0, 
        M_trials: int = 50, 
        total_shot_budget: float | int = 1e5, 
        active_qubits_count: int = 10, 
        beta_val: float = 0.0
    ) -> None:
        """
        Runs a statistical variance benchmark at a single specific k-point.

        Compares the energy expectation estimation variance between the old QWC 
        protocol and the new CMP protocol over a fixed number of independent random 
        sampling trials given an identical total shot budget.

        :param k_point: The target wave vector.
        :param state_index: Index of the target orthogonal state.
        :param M_trials: Number of independent simulation trials to calculate standard deviation.
        :param total_shot_budget: The total number of shots allowed across all execution bases.
        :param active_qubits_count: Size of active qubit pool for numerical angle templates. The number of active qubits cant be higher than the size of the system.
        :param beta_val: Constant phase value applied to the ansatz layers.
        """
        print("\n" + "="*70)
        print(f"      STATISTICAL BENCHMARK AT k-POINT (STATE INDEX {state_index})")
        print("\n" + "="*70)

        class_ham, _ = self.config.hamiltonian(k_point, *self.hpar)
        offdiag = self.vqe_solver._offdiag_terms(class_ham)
        
        # Calculate QWC group depth to see how to divide up the shot budget for the old method
        unique_j_rows = len(set([j for j, l in offdiag]))
        num_circuits_old = 2 + unique_j_rows  
        
        # Shot allocation logic: Old splits across all groups, New splits evenly across 3 bases
        shots_per_circuit_old = int(total_shot_budget // num_circuits_old)
        shots_per_circuit_new = int(total_shot_budget // 3)

        print(f" ► Total Shared Budget      : {total_shot_budget} shots")
        print(f" ► Active Qubit Pool Depth  : {active_qubits_count} qubits")
        print(f" ► State Phase (Beta) Value : {beta_val}")
        print(f" ► Old QWC Split Allocation : {shots_per_circuit_old} shots/circuit across {num_circuits_old} bases")
        print(f" ► New CMP Split Allocation : {shots_per_circuit_new} shots/circuit across 3 bases\n")

        fixed_history, target_angles = self.vqe_solver._generate_angles(
            n=state_index, 
            active_qubits_count=active_qubits_count, 
            beta_val=beta_val
        )
        ansatz_circuit, _ = self.vqe_solver._build_ansatz_circuit(
            n=state_index, 
            prev_params_list=fixed_history, 
            current_params=target_angles
        )
        transpiled_circuit = transpile(ansatz_circuit, simulator)

        print("Step 1/3: Computing exact analytical statevector reference...")
        exact_energy = self.vqe_solver.cost_func_oa_statevector([], transpiled_circuit, class_ham, offdiag)
        
        from qiskit.quantum_info import Statevector as QiskitSV
        state_data = QiskitSV.from_instruction(transpiled_circuit).data
        shot_probs, _, _, _ = cmp.amplitudes(transpiled_circuit, shots_per_circuit_new)
        
        print("\n" + "="*70)
        print("        SINGLE-EXCITATION PROBABILITY AMPLITUDES      ")
        print("\n" + "="*70)
        print(f"{'State/Qubit':<14} | {'Exact Statevector (|a_i|^2)':<28} | {f'Shot Simulator ({shots_per_circuit_new} Shots)':<25}")
        print("\n" + "="*70)
        
        for q_idx in range(self.num_qubits):
            state_dec_idx = 1 << q_idx
            bin_str = f"{state_dec_idx:0{self.num_qubits}b}"
            exact_prob = abs(state_data[state_dec_idx])**2
            sampled_prob = shot_probs[q_idx]
            
            print(f"|{bin_str}> (Q{q_idx}) | {exact_prob:<28.5f} | {sampled_prob:<25.5f}")
            
        print("\n" + "="*70)

        original_instance_shots = self.vqe_solver.Nshots
        
        print(f"Step 2/3: Evaluating Old QWC Protocol ({M_trials} iterations)...")
        self.vqe_solver.Nshots = shots_per_circuit_old
        old_energies = []
        for _ in range(M_trials):
            e_old = self.vqe_solver.cost_func_oa_old([], transpiled_circuit, class_ham, offdiag)
            old_energies.append(e_old)
        
        mean_old = np.mean(old_energies)
        std_old = np.std(old_energies, ddof=1)
        
        print(f"Step 3/3: Evaluating New CMP Protocol ({M_trials} iterations)...")
        self.vqe_solver.Nshots = shots_per_circuit_new
        new_energies = []
        for _ in range(M_trials):
            e_new = self.vqe_solver.cost_func_oa_new([], transpiled_circuit, class_ham, offdiag)
            new_energies.append(e_new)
            
        mean_new = np.mean(new_energies)
        std_new = np.std(new_energies, ddof=1)

        self.vqe_solver.Nshots = original_instance_shots
        
        print("\n" + "="*70)
        print("                    FINAL BENCHMARK DATA AT k-POINT                ")
        print("\n" + "="*70)
        print(f"EXACT STATEVECTOR ENERGY:                {exact_energy:.6f}")
        print("-" * 70)
        print(f"Old Protocol Mean Energy:                {mean_old:.6f}  (Error: {mean_old - exact_energy:+.6f})")
        print(f"New Protocol Mean Energy:                {mean_new:.6f}  (Error: {mean_new - exact_energy:+.6f})")
        print("-" * 70)
        print(f"Old Protocol Absolute Error:             {abs(exact_energy - mean_old):.6f}")
        print(f"New Protocol Absolute Error:             {abs(exact_energy - mean_new):.6f}")
        print("-" * 70)
        print(f"Old Protocol Standard Deviation (s_QWC): {std_old:.6f}")
        print(f"New Protocol Standard Deviation (s_New): {std_new:.6f}")
        print("-" * 70)
        if std_new > 0:
            print(f"Variance Reduction Factor (VRF):         {(std_old**2)/(std_new**2):.2f}x")
        print("\n" + "="*70)

    def statistical_benchmark_ksweep(
        self, 
        state_index: int = 0, 
        M_trials: int = 50, 
        total_shot_budget: float | int = 1e4, 
        active_qubits_count: int = 4, 
        beta_val: float = 0.0, 
        output_filename: str = "statistical_benchmark_ksweep_random.txt"
    ) -> None:
        """
        Sweeps across the full experiment path (path_q) to evaluate tracking metrics for both protocols.

        Saves detailed trial parameters, errors, variances, and calculated Variance Reduction Factors 
        (VRF) for every single wave vector k coordinate directly into a .txt file.

        :param state_index: Index of the target orthogonal state.
        :param M_trials: Number of sampling iterations executed per k-point.
        :param total_shot_budget: Total cumulative shot budget per k-point coordinate.
        :param active_qubits_count: Size of active qubit pool used for creating numerical ansatz layers.
        :param beta_val: Constant phase value applied across ansatz layers.
        :param output_filename: The name of the file to save report metrics to.
        """
        with open(output_filename, "w") as f:
            f.write("=============================================================================================================================================\n")
            f.write("                                                              STATISTICAL BENCHMARK METRICS k-SWEEP RUN                                      \n")
            f.write("=============================================================================================================================================\n")
            f.write(f"TOTAL SHOT BUDGET ALLOCATED PER K-POINT AND STATE: {int(total_shot_budget)} shots ({M_trials} trials per protocol)\n")
            f.write(f"STATE INDEX: {state_index} | ACTIVE QUBIT SUBSPACE POOL: {active_qubits_count} | (BETA): {beta_val}\n")
            f.write("---------------------------------------------------------------------------------------------------------------------------------------------\n")
            header = f"{'k_point [kx, ky]':<26} | {'Exact Energy':<13} | {'Mean Old':<11} | {'Mean New':<11} | {'Abs Err Old':<10} | {'Abs Err New':<10} | {'Std Old':<9} | {'Std New':<9} | {'Var Old':<9} | {'Var New':<9} | {'VRF':<7}\n"
            f.write(header)
            f.write("-" * len(header) + "\n") 

            # Use this if config_Si:     header = f"{'k_point [kx, ky, kz]':<26} 
            
        print(f"File created: '{output_filename}'. Starting k-sweep...")
        original_instance_shots = self.vqe_solver.Nshots
        
        for k_idx, k_point in enumerate(self.path_q):
            k_str = f"[{k_point[0]:.3f}, {k_point[1]:.3f}]" # change this to f"[{k_point[0]:.3f}, {k_point[1]:.3f}, {k_point[2]:.3f} ]" is config_Si
            print(f"\nProcessing k-point index {k_idx+1}/{len(self.path_q)}: k = {k_str}")
            
            class_ham, _ = self.config.hamiltonian(k_point, *self.hpar)
            offdiag = self.vqe_solver._offdiag_terms(class_ham)
            
            unique_j_rows = len(set([j for j, l in offdiag]))
            num_circuits_old = 2 + unique_j_rows
            
            shots_per_circuit_old = int(total_shot_budget // num_circuits_old)
            shots_per_circuit_new = int(total_shot_budget // 3)
            
            fixed_history, target_angles = self.vqe_solver._generate_angles(
                n=state_index, 
                active_qubits_count=active_qubits_count, 
                beta_val=beta_val
            )
            ansatz_circuit, _ = self.vqe_solver._build_ansatz_circuit(
                n=state_index, 
                prev_params_list=fixed_history, 
                current_params=target_angles
            )
            transpiled_circuit = transpile(ansatz_circuit, simulator)
            
            exact_energy = self.vqe_solver.cost_func_oa_statevector([], transpiled_circuit, class_ham, offdiag)
            
            # --- (OLD PROTOCOL RUN) ---
            self.vqe_solver.Nshots = shots_per_circuit_old
            old_energies = []
            for _ in range(M_trials):
                e_old = self.vqe_solver.cost_func_oa_old([], transpiled_circuit, class_ham, offdiag)
                old_energies.append(e_old)
            
            mean_old = np.mean(old_energies)
            std_old = np.std(old_energies, ddof=1)
            var_old = std_old ** 2
            err_old = mean_old - exact_energy
            
            # --- (NEW PROTOCOL RUN) ---
            self.vqe_solver.Nshots = shots_per_circuit_new
            new_energies = []
            for _ in range(M_trials):
                e_new = self.vqe_solver.cost_func_oa_new([], transpiled_circuit, class_ham, offdiag)
                new_energies.append(e_new)
                
            mean_new = np.mean(new_energies)
            std_new = np.std(new_energies, ddof=1)
            var_new = std_new ** 2
            err_new = mean_new - exact_energy
            vrf = (var_old / var_new) if var_new > 0 else 0.0
            
            data_line = (
                f"{k_str:<26} | "
                f"{exact_energy:<13.6f} | "
                f"{mean_old:<11.6f} | "
                f"{mean_new:<11.6f} | "
                f"{abs(err_old):<+10.6f} | "
                f"{abs(err_new):<+10.6f} | "
                f"{std_old:<9.6f} | "
                f"{std_new:<9.6f} | "
                f"{var_old:<9.6f} | "
                f"{var_new:<9.6f} | "
                f"{vrf:.2f}x\n"
            )
            
            with open(output_filename, "a") as f:
                f.write(data_line)
                
            print(f"Saved metrics for k = {k_str}. VRF = {vrf:.2f}x")
            
        self.vqe_solver.Nshots = original_instance_shots
        
        with open(output_filename, "a") as f:
            f.write("=============================================================================================================================================\n")
            
        print(f"\nSweep complete! All results saved successfully to '{output_filename}'")


    def run_eigenstate_angles(self, material_name: str | None = None) -> None:
        """
        Runs an exact statevector OAVQE routine across all k-points to extract 
        the true optimal ansatz angles for every individual eigenstate.

        Temporarily replaces the solver's execution method with an exact noiseless 
        'statevector' evaluation and minimises energy profiles using the L-BFGS-B 
        optimiser with high tight tolerance thresholds. Saves parameters to an HDF5 database.

        :param material_name: Optional string identifier to label the output HDF5 database.
        """
        if material_name is None:
            try:
                material_name = self.config.MATERIAL.lower().replace(" ", "_")
            except AttributeError:
                material_name = "unknown_material"
        else:
            material_name = material_name.lower().replace(" ", "_")
            
        output_filename = f"optimised_angles_{material_name}.h5"

        original_method_name = self.vqe_solver.method_name
        original_cost_func = self.vqe_solver.cost_func
        self.vqe_solver.method_name = 'statevector'
        self.vqe_solver.cost_func = self.vqe_solver.cost_func_oa_statevector
        
        target_states_count = self.num_qubits 
        
        print("\n" + "="*70)
        print(f"      EXACT STATEVECTOR OAVQE ({target_states_count} TOTAL STATES)")
        print("\n" + "="*70)
        print(f"Target Material    : {material_name.upper()}")
        print(f"Target Output Path : {output_filename}")
        print(f"Sweeping through {len(self.path_q)} k-points...\n")
        
        previous_k_angles = {}
        
        with h5py.File(output_filename, "w") as h5_file:
            h5_file.attrs["material"] = material_name
            h5_file.attrs["num_qubits"] = self.num_qubits
            h5_file.attrs["calculation_method"] = self.calc_method
            h5_file.attrs["timestamp"] = str(datetime.datetime.now())
            
            for k_idx, k_point in enumerate(self.path_q):
                k_str = f"[{k_point[0]:.3f}, {k_point[1]:.3f}]" # change this to f"[{k_point[0]:.3f}, {k_point[1]:.3f}, {k_point[2]:.3f} ]" is config_Si
                print(f"\n[k-point {k_idx+1}/{len(self.path_q)}] Optimising for k = {k_str}")
                
                k_group = h5_file.create_group(f"k_point_{k_idx}")
                k_group.create_dataset("coordinates", data=k_point)
                k_group.attrs["string_representation"] = k_str
                
                opt_parameters_list = []
                class_ham, _ = self.config.hamiltonian(k_point, *self.hpar)
                offdiag_terms = self.vqe_solver._offdiag_terms(class_ham)
                
                for n in range(target_states_count):
                    print(f"  └─ Optimising State Index {n}...")
                    
                    ansatz_template, param_vec = self.vqe_solver._build_ansatz_circuit(n, opt_parameters_list)
                    transpiled_template = transpile(ansatz_template, simulator)
                    num_var_params = len(param_vec) if param_vec is not None else 0
                    
                    if (k_idx > 0) and (self.bootstrapping) and (n in previous_k_angles):
                        initial_parameters = previous_k_angles[n]
                    else:
                        initial_parameters = np.random.uniform(0, 2 * np.pi, num_var_params)
                        
                    minimize_args = (transpiled_template, class_ham, offdiag_terms)
                    
                    if num_var_params > 0:
                        result = minimize(
                            self.vqe_solver.cost_func,
                            x0=initial_parameters,
                            args=minimize_args,
                            method='L-BFGS-B',
                            options={'maxiter': self.max_iter},
                            tol=1e-9
                        )
                        optimized_angles = result.x
                    else:
                        optimized_angles = np.array([])
                        
                    opt_parameters_list.append(optimized_angles)
                    previous_k_angles[n] = optimized_angles
                    state_dataset_name = f"state_{n}_angles"
                    k_group.create_dataset(state_dataset_name, data=optimized_angles)
                    
                print(f"Successfully processed and saved all state angles for k = {k_str}")
                
        self.vqe_solver.method_name = original_method_name
        self.vqe_solver.cost_func = original_cost_func
        
        print(f"\nExperiment complete! All configurations compiled into '{output_filename}'")

    def statistical_benchmark_ksweep_eigenstates(
        self, 
        material_name: str | None = None, 
        M_trials: int = 50, 
        total_shot_budget: float | int = 1e3
    ) -> None:
        """
        Loads optimised parameter profiles from a database to benchmark variance 
        performance across all k-points and eigenstates.

        Computes mean absolute errors, variances, and standard deviation distributions 
        comparing the baseline old QWC and new CMP shot allocation frameworks.

        :param material_name: Optional identity token lookup flag matching the target database.
        :param M_trials: Number of random evaluation trials executed per state.
        :param total_shot_budget: Total cumulative shot budget per k-point coordinate and state.
        """
        import os
        
        if material_name is None:
            try:
                material_name = self.config.MATERIAL.lower().replace(" ", "_")
            except AttributeError:
                material_name = "unknown_material"
        else:
            material_name = material_name.lower().replace(" ", "_")
            
        input_h5_filename = f"optimised_angles_{material_name}.h5"
        output_txt_filename = f"statistical_benchmark_ksweep_eigenstates_{material_name}.txt"
        
        if not os.path.exists(input_h5_filename):
            raise FileNotFoundError(
                f"Could not locate the optimised parameter file: '{input_h5_filename}'. "
                f"Please run the angle extraction experiment first."
            )

        with open(output_txt_filename, "w") as f:
            f.write("=============================================================================================================================================\n")
            f.write(f"                                                              STATISTICAL BENCHMARK METRICS LOADED: {material_name.upper()} (ALL STATES PATH)               \n")
            f.write("=============================================================================================================================================\n")
            f.write(f"TOTAL SHOT BUDGET ALLOCATED PER EVALUATION: {int(total_shot_budget)} shots ({M_trials} trials per protocol)\n")
            f.write(f"SOURCE WAVEFUNCTION GEOMETRY DATABASE    : {input_h5_filename}\n")
            f.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------\n")
            header = f"{'k_point [kx, ky]':<20} | {'State':<5} | {'Exact Energy':<13} | {'Mean Old':<11} | {'Mean New':<11} | {'Abs Err Old':<10} | {'Abs Err New':<10} | {'Std Old':<9} | {'Std New':<9} | {'Var Old':<9} | {'Var New':<9} | {'VRF':<7}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            # Use this if config_Si:     header = f"{'k_point [kx, ky, kz]':<26}
            
        print(f"Benchmark initiated: '{output_txt_filename}'")
        original_instance_shots = self.vqe_solver.Nshots
        
        with h5py.File(input_h5_filename, "r") as h5_data:
            num_saved_k = len([key for key in h5_data.keys() if key.startswith("k_point_")])
            
            for k_idx in range(num_saved_k):
                k_group_key = f"k_point_{k_idx}"
                k_group = h5_data[k_group_key]
                
                k_point = k_group["coordinates"][:]
                k_str = k_group.attrs["string_representation"]
                
                print(f"\nEvaluating System Coordinate Block {k_idx+1}/{num_saved_k}: k = {k_str}")
                
                class_ham, _ = self.config.hamiltonian(k_point, *self.hpar)
                offdiag = self.vqe_solver._offdiag_terms(class_ham)
                
                unique_j_rows = len(set([j for j, l in offdiag]))
                num_circuits_old = 2 + unique_j_rows
                
                shots_per_circuit_old = int(total_shot_budget // num_circuits_old)
                shots_per_circuit_new = int(total_shot_budget // 3)
                
                opt_parameters_list = []
                
                for n in range(self.num_qubits):
                    state_key = f"state_{n}_angles"
                    target_angles = k_group[state_key][:]
                    
                    print(f"  ├─ Processing State Index {n}...")
                    
                    ansatz_circuit, _ = self.vqe_solver._build_ansatz_circuit(
                        n=n, 
                        prev_params_list=opt_parameters_list, 
                        current_params=target_angles
                    )
                    transpiled_circuit = transpile(ansatz_circuit, simulator)
                    
                    exact_energy = self.vqe_solver.cost_func_oa_statevector([], transpiled_circuit, class_ham, offdiag)
                    
                    # --- (OLD PROTOCOL RUN) ---
                    self.vqe_solver.Nshots = shots_per_circuit_old
                    old_energies = []
                    for _ in range(M_trials):
                        e_old = self.vqe_solver.cost_func_oa_old([], transpiled_circuit, class_ham, offdiag)
                        old_energies.append(e_old)
                    
                    mean_old = np.mean(old_energies)
                    std_old = np.std(old_energies, ddof=1)
                    var_old = std_old ** 2
                    err_old = mean_old - exact_energy
                    
                    # --- (NEW CMP PROTOCOL RUN) ---
                    self.vqe_solver.Nshots = shots_per_circuit_new
                    new_energies = []
                    for _ in range(M_trials):
                        e_new = self.vqe_solver.cost_func_oa_new([], transpiled_circuit, class_ham, offdiag)
                        new_energies.append(e_new)
                        
                    mean_new = np.mean(new_energies)
                    std_new = np.std(new_energies, ddof=1)
                    var_new = std_new ** 2
                    err_new = mean_new - exact_energy
                    
                    vrf = (var_old / var_new) if var_new > 0 else 0.0
                    
                    data_line = (
                        f"{k_str:<20} | "
                        f"{n:<5} | "
                        f"{exact_energy:<13.6f} | "
                        f"{mean_old:<11.6f} | "
                        f"{mean_new:<11.6f} | "
                        f"{abs(err_old):<+10.6f} | "
                        f"{abs(err_new):<+10.6f} | "
                        f"{std_old:<9.6f} | "
                        f"{std_new:<9.6f} | "
                        f"{var_old:<9.6f} | "
                        f"{var_new:<9.6f} | "
                        f"{vrf:.2f}x\n"
                    )
                    
                    with open(output_txt_filename, "a") as f:
                        f.write(data_line)
                    
                    opt_parameters_list.append(target_angles)
                    
                print(f"Completed benchmarking all eigenstates for k = {k_str}")
                
        self.vqe_solver.Nshots = original_instance_shots
        
        with open(output_txt_filename, "a") as f:
            f.write("=============================================================================================================================================\n")
            
        print(f"\nAnalysis complete! Complete dataset compiled into '{output_txt_filename}'")