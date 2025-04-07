# kuramoto_solver.py
import numpy as np
import dedalus.public as d3
import logging

logger = logging.getLogger(__name__)

def run_kuramoto_sim(L=22, N=64, stop_time=10, timestep=1e-1):
    # Define coordinate, basis, distributor
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=np.complex128)
    xbasis = d3.ComplexFourier(xcoord, size=N, bounds=(0, L), dealias=1)

    # Define field and equation
    u = dist.Field(name='u', bases=xbasis)
    dx = lambda A: d3.Differentiate(A, xcoord)

    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("dt(u) + dx(dx(u)) + dx(dx(dx(dx(u)))) = -u*dx(u)")

    # Initial condition
    x = dist.local_grid(xbasis)
    u['g'] = np.cos(2 * np.pi * x / L)

    # Solver
    solver = problem.build_solver(d3.RK443)
    solver.stop_sim_time = stop_time

    u_store_grid = [np.copy(u['g'].flatten())]
    u_store_coeff = [np.copy(u['c'])]
    T_store = [solver.sim_time]

    try:
        while solver.proceed:
            solver.step(timestep)
            u_store_grid.append(u['g'].flatten())  # ensures shape (N,)
            u_store_coeff.append(np.copy(u['c']))
            T_store.append(solver.sim_time)

            if solver.iteration % 100 == 0:
                logger.info(f"Iteration = {solver.iteration}, time = {solver.sim_time:.4f}")
    except Exception as e:
        logger.error("Simulation error:", exc_info=True)
        raise e
    finally:
        solver.log_stats()

    return np.array(u_store_grid), np.array(T_store)
