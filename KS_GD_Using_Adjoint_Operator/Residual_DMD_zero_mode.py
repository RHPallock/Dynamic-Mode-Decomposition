import numpy as np
import matplotlib.pyplot as plt
import h5py

# Load actual solution (Dedalus output)
with h5py.File("Actual_Solution/full_solution_s1.h5", "r") as f:
    u_actual_all = f["tasks/u"][:]  # shape: (time_steps, 1, N)
    x_actual = f["scales/x_hash_6e1dfcbb09985d14622b73ca81ef2c00e8d954ad"][:]  # grid points
    t_actual = f["scales/sim_time"][:]

# Extract the last snapshot
u_actual = u_actual_all[-1,:]  # shape: (N,)
u_actual /= np.max(np.abs(u_actual))  # normalize


# Load DMD predicted mode
with h5py.File("dmd_steady_mode.h5", "r") as f:
    u_dmd = f["steady_mode"][:]     # shape: (N,)
    x_dmd = f["x"][:]
    omega_dmd = f["omega_closest_to_zero"][()]

u_dmd /= np.max(np.abs(u_dmd))  # normalize


#Plot comparison
plt.figure(figsize=(8, 5))
plt.plot(x_actual, u_actual, label='Actual Final Solution', linewidth=2)
plt.plot(x_dmd, u_dmd, '--', label='DMD Approximate Steady Mode', linewidth=2)
plt.title("Comparison: Final KS Snapshot vs DMD Predicted Steady Mode")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("compare_actual_vs_dmd.png")
plt.show()


# Print L2 norm 
l2_norm = np.linalg.norm(u_actual - u_dmd)
print(f"L2 norm of the difference between actual and DMD mode: {l2_norm}")
