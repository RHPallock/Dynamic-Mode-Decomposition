import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import svd, eig
from kuramoto_solver import run_kuramoto_sim

# --------------------------
# 1. Simulate full solution
# --------------------------
u_store_grid, T_store = run_kuramoto_sim(stop_time=10)
U_data = u_store_grid.T  # shape: (space, time)
T_store = np.array(T_store)
dt = T_store[1] - T_store[0]

# --------------------------
# 2. Use only first 10 sec for DMD
# --------------------------
T_cutoff = 10  # seconds
idx_cut = np.argmax(T_store >= T_cutoff)

X = U_data[:, :idx_cut-1]
X_prime = U_data[:, 1:idx_cut]

# --------------------------
# 3. Perform DMD on training data
# --------------------------
U, S, Vh = svd(X, full_matrices=False)
r = 20
Ur = U[:, :r]
Sr = np.diag(S[:r])
Vr = Vh[:r, :]

A_tilde = Ur.T @ X_prime @ Vr.T @ np.linalg.inv(Sr)
eigvals, W = eig(A_tilde)
Phi = X_prime @ Vr.T @ np.linalg.inv(Sr) @ W
omega = np.log(eigvals) / dt
b = np.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]

# --------------------------
# 4. Predict full time range using DMD
# --------------------------
time_dynamics = np.zeros((r, len(T_store)), dtype=complex)
for i, t in enumerate(T_store):
    time_dynamics[:, i] = b * np.exp(omega * t)

U_dmd = (Phi @ time_dynamics).real  # shape: (space, time)

# --------------------------
# 5. Plot DMD Error Growth
# --------------------------
errors = np.linalg.norm(U_data - U_dmd, axis=0)
plt.figure(figsize=(6, 4))
plt.plot(T_store, errors)
plt.xlabel('Time (s)')
plt.ylabel('||u_actual - u_DMD||')
plt.title('DMD Approximation Error Over Time')
plt.grid(True)
plt.tight_layout()
plt.savefig('dmd_error_vs_time.png')
plt.show()

print("Error plot saved as dmd_error_vs_time.png")

# --------------------------
# 6. Create side-by-side video
# --------------------------
x = np.linspace(0, 22, U_data.shape[0])  # spatial grid from L=22

fig, ax = plt.subplots(figsize=(8, 4))
line1, = ax.plot(x, U_data[:, 0], label='True u(x,t)')
line2, = ax.plot(x, U_dmd[:, 0], '--', label='DMD Approx')
ax.set_ylim(-2, 2)
ax.set_xlabel('x')
ax.set_ylabel('u(x, t)')
ax.set_title('DMD vs True: Kuramoto–Sivashinsky')
ax.legend()
ax.grid(True)

def update(frame):
    line1.set_ydata(U_data[:, frame])
    line2.set_ydata(U_dmd[:, frame])
    ax.set_title(f't = {T_store[frame]:.2f} s')
    return line1, line2

ani = animation.FuncAnimation(fig, update, frames=len(T_store), blit=True)

# Try to save as .mp4 if ffmpeg is available, otherwise fallback to .gif
try:
    ani.save("dmd_vs_true_prediction.mp4", fps=10, writer='ffmpeg')
    print("🎥 Video saved as dmd_vs_true_prediction.mp4")
except Exception as e:
    print("⚠️ ffmpeg not found, saving as .gif instead")
    ani.save("dmd_vs_true_prediction.gif", fps=10)
    print("🎥 Video saved as dmd_vs_true_prediction.gif")
