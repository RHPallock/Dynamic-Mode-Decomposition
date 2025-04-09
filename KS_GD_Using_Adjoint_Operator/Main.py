import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.linalg import svd, eig
from ks_equilibrium_solver import run_ks_equilibrium_solver

# --------------------------
# last 50 snapshots
# --------------------------
u_store_grid, T_store, residue = run_ks_equilibrium_solver(residue_tol=1e-2, n_snapshots=50)
U_data = u_store_grid.T  # shape: (space, time)
T_store = np.array(T_store)
dt = T_store[1] - T_store[0]

# --------------------------
# DMD on entire 50 snapshots
# --------------------------
X = U_data[:, :-1]
X_prime = U_data[:, 1:]

U, S, Vh = svd(X, full_matrices=False)
r = min(20, len(S))  # Safe check if less than 20 snapshots
Ur = U[:, :r]
Sr = np.diag(S[:r])
Vr = Vh[:r, :]

A_tilde = Ur.T @ X_prime @ Vr.T @ np.linalg.inv(Sr)
eigvals, W = eig(A_tilde)
Phi = X_prime @ Vr.T @ np.linalg.inv(Sr) @ W
omega = np.log(eigvals) / dt
b = np.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]

# --------------------------
# Extract the steady mode (omega closest to 0)
# --------------------------
idx_closest = np.argmin(np.abs(omega))
steady_mode = Phi[:, idx_closest].real
steady_mode /= np.max(np.abs(steady_mode))  # normalize

# --------------------------
# Compare to final true simulation state
# --------------------------
true_final = U_data[:, -1]
true_final /= np.max(np.abs(true_final))  # normalize

x = np.linspace(0, 22, steady_mode.shape[0])
plt.figure(figsize=(7, 5))
plt.plot(x, true_final, label='Final snapshot from Adjoint', linewidth=2)
plt.plot(x, steady_mode, '--', label='DMD approx. equilibrium', linewidth=2)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Final KS Solution vs. DMD Approximate Equilibrium')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('final_vs_dmd_equilibrium.png')
plt.show()

# save the steady mode for further analysis in .h5 file

with h5py.File("dmd_steady_mode.h5", "w") as f:
    f.create_dataset("steady_mode", data=steady_mode)
    f.create_dataset("x", data=x)
    f.create_dataset("omega_closest_to_zero", data=omega[idx_closest].real)

print(" Comparison plot saved as 'final_vs_dmd_equilibrium.png'")
