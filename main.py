import argparse
import importlib
import sys

from oa_vqe import OrthogonalAnsatzVQE
from experiments import OAVQEExperiments

def main() -> None:

    parser = argparse.ArgumentParser(description="OAVQE Experiment Runner Suite")
    parser.add_argument("config_module", type=str, 
                        help="The name of the configuration file module (e.g., config_CuO2, config_bg or config_Si)")
    
    parser.add_argument("--exp", type=str, required=True,
                        choices=[
                            "run_oa_vqe", 
                            "run_eigenstates_angles", 
                            "run_benchmark_k", 
                            "run_benchmark_ksweep_eigenstates",
                            "run_benchmark_ksweep_random"
                        ],
                        help="The experiment routine to execute.")
    
    parser.add_argument("--method", type=str, default="COBYQA",
                        help="Optimisation algorithm for standard runs (default: COBYQA)")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Maximum optimisation iterations (default: 1000)")
    parser.add_argument("--no_bootstrap", action="store_true",
                        help="Disable parameter bootstrapping across points/states")
    parser.add_argument("--protocol", type=str, default="new", choices=["old", "new", "statevector"],
                        help="Measurement protocol to use for the VQE run (default: new)")
    
    parser.add_argument("--shots", type=float, default=1e4,
                        help="Total shared shot budget for benchmarks (default: 10000)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of macro trials per benchmark (default: 50)")
    
    parser.add_argument("--k_idx", type=int, default=0,
                        help="Index of the k-point to target for single-point benchmarks (default: 0)")
    parser.add_argument("--state_idx", type=int, default=0,
                        help="The excited state index to target inside benchmarks (default: 0)")
    parser.add_argument("--active_qubits", type=int, default=4,
                        help="Active qubit pool subspace depth (default: 4)")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="State phase beta angle shift value (default: 0.0)")

    args = parser.parse_args()

    print(f"Loading configuration pipeline from target: {args.config_module}.py...")
    try:
        config = importlib.import_module(args.config_module)
    except ModuleNotFoundError:
        print(f" Error: Could not locate file '{args.config_module}.py' in this directory.")
        sys.exit(1)

    print("Initialising Solver...")
    physics_solver = OrthogonalAnsatzVQE(config=config)
    
    suite = OAVQEExperiments(
        vqe_solver=physics_solver,
        calc_method=args.method,
        max_iter=args.max_iter,
        bootstrapping=not args.no_bootstrap,
        protocol=args.protocol
    )

    print(f"\n Routing to requested routine [{args.exp}]...")
    
    material_clean_name = args.config_module.replace("config_", "")

    if args.exp == "run_oa_vqe":
        suite.run_calculation()

    elif args.exp == "run_eigenstates_angles":
        suite.run_eigenstate_angles(material_name=material_clean_name)

    elif args.exp == "run_benchmark_k":
        if args.k_idx >= len(suite.path_q):
            print(f" Error: Requested k_idx {args.k_idx} is out of bounds (Path length: {len(suite.path_q)}).")
            sys.exit(1)
            
        target_k = suite.path_q[args.k_idx]
        suite.statistical_benchmark_kpoint(
            k_point=target_k,
            state_index=args.state_idx,
            M_trials=args.trials,
            total_shot_budget=args.shots,
            active_qubits_count=args.active_qubits,
            beta_val=args.beta
        )

    elif args.exp == "run_benchmark_ksweep_eigenstates":
        try:
            suite.statistical_benchmark_ksweep_eigenstates(
                material_name=material_clean_name,
                M_trials=args.trials,
                total_shot_budget=args.shots
            )
        except FileNotFoundError as e:
            print(f" {e}")
            print(" Hint: Run: python main.py [config] --exp run_eigenstates_angles first.")

    elif args.exp == "run_benchmark_ksweep_random":
        suite.statistical_benchmark_ksweep(
            state_index=args.state_idx,
            M_trials=args.trials,
            total_shot_budget=args.shots,
            active_qubits_count=args.active_qubits,
            beta_val=args.beta,
            output_filename=f"statistical_benchmark_ksweep_random_{material_clean_name}.txt"
        )

    print("\n Execution chain completed successfully.")

if __name__ == "__main__":
    main()