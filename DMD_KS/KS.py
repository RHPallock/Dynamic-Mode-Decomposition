# Rumayel Hassan Pallock
# Kuramoto sivanski equation solution using dedalus version 3.0
# Date 1/13/2023
# equation----- dt(u) = -u*dx(u) - dx(dx(u)) - dx(dx(dx(dx(u))))


import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import pathlib
import subprocess
import os
import h5py # HDF5 file manipulation
import scipy.io
from mpi4py import MPI
import time
import logging
logger = logging.getLogger(__name__)
#import shutil
#shutil.rmtree('full_solution',ignore_errors = True)

#Parameters
L = 39


# Define co-ordinate, distributor and basis
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord,dtype = np.complex128)
xbasis = d3.ComplexFourier(xcoord,size = 64,bounds = (0,L),dealias = 3/2)


# define fields 
# define field variable u
u = dist.Field (name='u', bases = xbasis)

# define the derivatives with substituions
dx = lambda A: d3.Differentiate(A,xcoord)

# Add equations
# main equation , with linear terms on lhs and nonlinear terms on rhs
problem = d3.IVP([u],namespace = locals())
problem.add_equation("dt(u)+dx(dx(u))+dx(dx(dx(dx(u)))) = -u*dx(u)")


# Initial condition
x = dist.local_grid(xbasis)
u['c'][0] = 0
u['c'][1] =0
u['c'][2] = 0
u['c'][3] = 0
u['c'][4] = 2.6
u['c'][10] = 0

# Build solver
solver = problem.build_solver(d3.RK443)

# Setting stop criteriatho
solver.stop_sim_time = 50000


# analysis
# Output solution fields
full_solution = solver.evaluator.add_file_handler('KS_Td', iter =1 , max_writes = 50000)
full_solution.add_task(u, layout = 'g', name='u')
full_solution.add_task(u, layout = 'c', name='u_coeff')




# main loop
timestep = 1e-1
try:
    while solver.proceed:
        solver.step(timestep)
        if solver.iteration % 100 == 0:
            logger.info('Iteration = %i, time = %e, dt = %e' %(solver.iteration, solver.sim_time, timestep))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

