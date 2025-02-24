# Navier-Stokes Solver using Finite Elements. A biomedical engineering project. #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Tue 21 Jan 2025 @ 18:02:45 +0100
# Modified: Sun 02 Feb 2025 @ 21:50:12 +0100

# Packages
from fenics import *
import numpy as np

def NSE(mu, rho, theta, dt, T, Re, R, ord=[2,1], backflow_stab=False, temam_stab=False, pressure_convection_stab=False):
    """ Navier-Stokes Finite Element Solver for Biomedical Engineering.
    mu:     viscocity of the fluid  |   (float)
    rho:    density of the fluid    |   (float)
    theta:  theta method parameter  |   (float)
    dt:     time-step               |   (float)
    T:      total simulation uptime |   (float)
    Re:     Reynolds number         |   (float)
    R:      inlet velocity parameter|   (float)
    ord:    element degree          |   (list)
    """

    # Define Mesh & Boundaries
    # Custom geometry & boundaries, simulating 60% stenosis of a blood vessel
    mesh = Mesh("stenosis_f0.6_fine.xml")
    boundaries = MeshFunction("size_t", mesh, "stenosis_f0.6_fine_facet_region.xml")

    # Define parameters as constants in Fenics
    mu = Constant(mu)
    rho = Constant(rho)
    theta = Constant(theta)

    # Define Spaces (piecewise polynomials)
    VE = VectorElement("P", mesh.ufl_cell(), ord[0])
    PE = FiniteElement("P", mesh.ufl_cell(), ord[1])
    F = FunctionSpace(mesh, VE * PE)

    # Define custom measurement
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Define Trial & Test Functions
    u, p = TrialFunctions(F)        # u_{n+1}, p_{n+1}
    v, q = TestFunctions(F)
    w = Function(F)
    u_n, p_n = w.split()            # u_{n}, p_{n}

    # Define theta method parameters
    u_kpt = theta * u + (1 - theta) * u_n
    p_kpt = theta * p + (1 - theta) * p_n

    # Define weak form constants
    k = Constant(1 / dt)
    n = FacetNormal(mesh)
    C_back = Constant(5)

    # Define Weak Form
    A = (k * rho * dot(u - u_n, v) * dx
         + rho * dot(grad(u_kpt)*u_n, v) * dx
         + mu * inner(grad(u_kpt), grad(v)) * dx
         - p_kpt * div(v) * dx
         + q * div(u) * dx)

    # Define & Introduce Backflow Stabilization
    if backflow_stab == True:
        u_dot_n = dot(u_n, n)
        u_neg = 0.5 * abs(u_dot_n - abs(u_dot_n))
        u_back = C_back * u_neg * dot(u_kpt, v) * ds(2)
        A += u_back
    else:
        pass

    # Define & Introduce Temam Stabilization
    if temam_stab == True:
        temam = (rho / 2) * dot(div(u_n), dot(u_kpt, v)) * dx
        A += temam
    else:
        pass

    # Define & Introduce Joint Pressuse-Convection Stabilization
    if pressure_convection_stab == True:
        h = CellDiameter(mesh)
        delta_1 = 4 / (dt**2)
        delta_2 = (4 * dot(u_n, u_n)**2) / (h**2)
        delta_3 = ((12 * mu) / (rho * (h**2)))**2
        delta = 1.0 / sqrt(delta_1 + delta_2 + delta_3)
        joint_stabilization = delta * (dot(rho * grad(u_kpt)*u_n + grad(p_kpt), rho * grad(v)*u_n + grad(q))) * dx
        A += joint_stabilization
    else:
        pass

    # Define Boundary Parameters and Inlet Velocity
    U_bulk = float((Re * mu) / (2 * rho * R))
    inlet_velocity = Expression(("1.5 * U_bulk * sin(DOLFIN_PI * t * 2.5) * (1.0 - pow(x[1]/R, 2))", "0.0"), degree = 5, U_bulk = U_bulk, t = 0.0, T = T, R = R)

    # Define Boundary Conditions
    bc_inflow = DirichletBC(F.sub(0), inlet_velocity, boundaries, 1)
    bc_symmetry = DirichletBC(F.sub(0).sub(1), Constant(0.0), boundaries, 3)
    bc_noslip = DirichletBC(F.sub(0), Constant((0.0, 0.0)), boundaries, 4)
    bcs = [bc_inflow, bc_symmetry, bc_noslip]

    # Split weak form
    a = lhs(A)
    L = rhs(A)
    K = assemble(a)

    # Rename parameters for paraview
    u_n.rename("u", "u")
    p_n.rename("p", "p")

    # Initialize xdmf files
    xdmf_u = XDMFFile("results/u.xdmf")
    xdmf_p = XDMFFile("results/p.xdmf")
    
    # Time loop
    t = 0.0
    while t < T:
        # Assemble problem
        assemble(a, tensor=K)
        b = assemble(L)
        [bc.apply(K, b) for bc in bcs]
        # Update time-dependent boundary condition
        inlet_velocity.t = t
        # Solve weak form
        solve(K, w.vector(), b)
        # Write output on files
        xdmf_u.write(u_n, t)
        xdmf_p.write(p_n, t)
        t += dt

    del xdmf_u
    del xdmf_p

    print("Simulation completed successfully.")

def main():
    """ Simple CLI for Navier-Stokes Solver """
    mu = float(input("Please give the viscocity value of the fluid: "))
    rho = float(input("Please give the density value of the fluid: "))
    theta = float(input("Please give the theta parameter to define explicit, semi-explicit, implicit time integration scheme: "))
    dt = float(input("Please define an acceptable time step for the experiment: "))
    T = float(input("Please define the total simulation uptime (seconds): "))
    Re = float(input("Please give the Reynolds number: "))
    R = float(input("Please give the inlet velocity parameter: "))
    ord1 = int(input("Please define the piecewise polynomial element degree for the velocity space: "))
    ord2 = int(input("Please define the piecewise polynomial element degree for the pressure space: "))
    order = [ord1, ord2]
    backflow = input("Do you want to penalize negative velocities at the outflow boundary by introducing backflow stabilization (y/n)? ").lower().strip() == "y"
    temam = input("Do you want to introduce Temam stabilization, balancing out the excess numerical energy of the simulation (y/n)? ").lower().strip() == "y"
    joint = input("Do you want to introduce pressure and convection stabilization, smoothing out numerical errors (y/n)? ").lower().strip() == "y"
    print("Running, this might take some time.")
    NSE(mu, rho, theta, dt, T, Re, R, ord=order, backflow_stab=backflow, temam_stab=temam, pressure_convection_stab=joint)
    print("Velocity and Pressure solutions stored in 'results/' folder. Visualize them using Paraview.")

main()
