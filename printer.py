import matplotlib.pyplot as plt
import numpy as np
import h5py
import tkinter as tk
from tkinter import filedialog
from datetime import timedelta

root = tk.Tk()
root.withdraw()

class QBSPrinter():
    def __init__(self):
        self.load_file()

    def load_file(self):
        file_path = filedialog.askopenfilename(
        title="Choose .h5 file",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        print("Selected file:", file_path)
        with h5py.File(file_path, "r") as f:
            print(f"The method used in the minimisation was {f['calc_method'][()].decode('utf-8')}.")
            print(f"Bootstrapping was{' NOT' if not f['bootstrapping'][()] else ''} used in the computation.")
            print(f"The computation took {timedelta(seconds=f['calculated_values_duration'][()])} h:m:s.")

            kpoint_group_names = sorted([
            key for key in f.keys()
            if key.startswith('k-point index')
            ], key=lambda x: int(x.split(' ')[-1]))
            
            self.eigenvalues_per_kpoint = []
            self.n_fun_per_kpoint = []
            self.optimal_params_per_kpoint = []
            self.minimize_time_per_kpoint = []
            self.kpoint_indices = []

            for group_name in kpoint_group_names:
                
                k_index = int(group_name.split(' ')[-1])
                self.kpoint_indices.append(int(k_index))
                self.eigenvalues_per_kpoint.append(f[group_name]['eigenvalues'][()])
                self.n_fun_per_kpoint.append(f[group_name]['n_fun'][()])
                self.optimal_params_per_kpoint.append(f[group_name]['optimal_params'][()])
                self.minimize_time_per_kpoint.append(f[group_name]['minimize_time'][()])

            self.exact_eigenvalues = f['exact_values']['eigenvalues'][()]
            self.exact_path = f['exact_values']['path'][()]
            self.calc_path_print = f['path_print'][()]
            self.labels_name = list(f['labels_name'].asstr()[...])
            self.labels_position = f['labels_position'][()]

    def plot_eigenvalues(self, choosed_color="green"):
        """
        Plots of the OAVQE energies and the exact energies in one graph.
        """

        fig, ax = plt.subplots()
        ax.plot(self.exact_path, self.exact_eigenvalues, color="gray")
        ax.plot(self.calc_path_print, self.eigenvalues_per_kpoint, 'o', color=choosed_color, mfc="none", markersize=4, markeredgewidth=2)
        ax.set_xticks(self.labels_position)
        ax.set_xticklabels(self.labels_name, fontsize=10)
        ax.set_xlabel('Wave vector k')
        ax.set_ylabel('Energy [eV]')
        ax.set_title('Calculated (colored circles) and exact (gray line) band structure')
        plt.tight_layout()
        plt.show()

    def plot_each_eigenvalues(self):
        """
        Plots the VQD energies and the exact energies in one graph, each state in different color.
        """

        num_eigenvalues_per_kpoint = self.eigenvalues_per_kpoint[0].shape[0]
        eigenvalues_transposed = np.array(self.eigenvalues_per_kpoint).T

        fig, ax = plt.subplots()
        ax.plot(self.exact_path, self.exact_eigenvalues, color="gray")

        for i in range(num_eigenvalues_per_kpoint):
            ax.plot(self.calc_path_print, eigenvalues_transposed[i], 'o', mfc="none", markersize=4, markeredgewidth=2, label=f'n = {i+1}')

        ax.set_xticks(self.labels_position)
        ax.set_xticklabels(self.labels_name, fontsize=10)
        ax.set_xlabel('Wave vector k')
        ax.set_ylabel('Energy [eV]')
        ax.set_title('Calculated (colored) and exact (gray line) band structure')
        plt.tight_layout() 
        plt.show()

    def plot_n_fun(self):
        """
        Plots the number of function evaluations for every energy level
        along the k-point path. Each energy (eigenvalue) is shown as a separate line.
        """
        fig, ax = plt.subplots()

        n_eigenvalues = len(self.n_fun_per_kpoint[0])  
        n_fun_array = np.array(self.n_fun_per_kpoint)  
        for i in range(n_eigenvalues):
            ax.plot(self.calc_path_print, n_fun_array[:, i], marker='o', label=f'Eigenvalue {i}')

        ax.set_xticks(self.labels_position)
        ax.set_xticklabels(self.labels_name, fontsize=15)

        ax.set_xlabel('Wave vector k', fontsize = 20)
        ax.set_ylabel(r'N$_{\text{f}}$', fontsize = 20)
    
        ax.legend(loc='center', bbox_to_anchor=(0.5, 0.25), fontsize=12)
        plt.tight_layout()
        plt.show()



    def plot_calc_time(self):
        """
        Plots the map of the minimisation time for each state and wave vector k.
        """

        fig, ax = plt.subplots(gridspec_kw={'height_ratios': [4]})
        fig.suptitle(f"Whole duration: {timedelta(seconds=np.sum(self.minimize_time_per_kpoint))} h:m:s", fontsize=14)
        data_2d = np.array(self.minimize_time_per_kpoint).T
        im1 = ax.imshow(data_2d, origin='lower', aspect='auto', cmap='viridis')
        fig.colorbar(im1, ax=ax)
        ax.set_xticks(self.labels_position)
        ax.set_xticklabels(self.labels_name, fontsize=10)
        ax.set_xlabel('Wave vector k')
        ax.set_ylabel('State')
        ax.set_yticks(np.arange(data_2d.shape[0]))
        #ax.set_xticks(np.arange(data_2d.shape[1]))
        ax.set_title(f"Duration in [s] for each state and k-point")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    s = QBSPrinter()
    s.plot_eigenvalues()
    s.plot_each_eigenvalues()
    s.plot_n_fun()
    s.plot_calc_time()
   
