import numpy as np
import dedalus.public as d3
import logging
import shutil

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_ks_equilibrium_solver(L=22, N=32, timestep=0.1, stop_time=4000, residue_tol=1e-10, n_snapshots=50):
    # Clean output folder
    shutil.rmtree('full_solution', ignore_errors=True)

    # Coordinates, basis, fields
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=np.float64)
    xbasis = d3.RealFourier(xcoord, size=N, bounds=(0, L), dealias=1)

    u = dist.Field(name='u', bases=xbasis)
    dx = lambda A: d3.Differentiate(A, xcoord)

    # RHS operator
    def RHS(u_b):
        ux = dx(u_b)
        uxx = dx(ux)
        uxxx = dx(uxx)
        uxxxx = dx(uxxx)
        return -u_b * ux - uxx - uxxxx

    def RHS_ev(u_b):
        return RHS(u_b).evaluate()

    # PDE definition
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation(
        "dt(u)+dx(dx(dx(dx(u))))+2*dx(dx(dx(dx(dx(dx(u)))))) + dx(dx(dx(dx(dx(dx(dx(dx(u)))))))) = "
        "u*dx(u*dx(u)+dx(dx(u))+dx(dx(dx(dx(u)))))-dx(dx(u*dx(u))) - dx(dx(dx(dx(u*dx(u)))))"
    )

    # Initial condition
    x = dist.local_grid(xbasis)
    u['g'] = 2 * np.sin(2 * np.pi * x / L)

    # Solver
    solver = problem.build_solver(d3.RK443)
    solver.stop_sim_time = stop_time

    # # Output handler
    # full_solution = solver.evaluator.add_file_handler('full_solution', iter=iter_save, max_writes=500000)
    # full_solution.add_task(u, layout='g', name='u')

    # Storage
    total_iteration = int(stop_time / timestep)
    L2_residue = np.zeros((total_iteration, 1))
    u_snapshots = []
    T_snapshots = []

    iteration = 0

    # Main loop
    while solver.proceed:
        solver.step(timestep)
        rhs_val = RHS_ev(u)['g']
        L2_norm = np.linalg.norm(rhs_val * rhs_val)
        L2_residue[iteration, 0] = L2_norm
        iteration += 1


        u_snapshots.append(u['g'].copy().flatten())
        T_snapshots.append(solver.sim_time)

        if solver.iteration % 100 == 0:
            logger.info(f"Iteration = {solver.iteration}, time = {solver.sim_time:.2f}, dt = {timestep}, residue = {L2_norm}")
        if L2_norm <= residue_tol:
            logger.info("Converged: L2 residue below threshold.")
            break

    # Get last n_snapshots
    u_snapshots = np.array(u_snapshots)
    T_snapshots = np.array(T_snapshots)

    if len(T_snapshots) < n_snapshots:
        raise ValueError(f"Only {len(T_snapshots)} snapshots available, but {n_snapshots} requested.")

    return (
        u_snapshots[-n_snapshots:],  # shape: (n_snapshots, N)
        T_snapshots[-n_snapshots:],  # shape: (n_snapshots,)
        L2_residue[:iteration+1]     # full L2 residue history
    )
