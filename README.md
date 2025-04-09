In this repo, Dynamic mode decomposition techniques has been applied on snapshots found from simulation Adjoint PDE of KS equations.

DMD_KS:   

        Run DMD_ks.py inside the folder
        - Simulates the KS equation for 10 seconds
        - Uses data of 10 seconds for decomposition of dynamics modes
        - Recreate first 10 seconds using the modes
        

KS_GD_Using_Adjoint_Operator

        Run Main.py inside the folder.
        - takes last 50 snapshots from the simulation of adjoint PDE of KS equation
        - Finds the Dynamics Modes
        - Finds the closest eigenvalue towards 0
        - Approximates steady mode

        Also inside, there's code for comparing with Actual equilibrium with the dmd approximated equilibrium.
        
