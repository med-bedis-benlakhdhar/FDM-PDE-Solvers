import numpy as np
import matplotlib.pyplot as plt

def solve_heat_2d_explicit(Nx, Ny, Nt, L, W, T, alpha):
    """
    Solves 2D heat equation: u_t = alpha * (u_xx + u_yy)
    Using Vectorized Explicit Euler.
    """
    dx = L / Nx
    dy = W / Ny
    dt = T / Nt
    
    # CFL Condition Check for 2D
    cfl_x = alpha * dt / (dx**2)
    cfl_y = alpha * dt / (dy**2)
    
    if (cfl_x + cfl_y) > 0.5:
        print(f"WARNING: Scheme unstable. dt is too large. CFL = {cfl_x + cfl_y:.3f} > 0.5")
    
    x = np.linspace(0, L, Nx + 1)
    y = np.linspace(0, W, Ny + 1)
    X, Y = np.meshgrid(x, y)
    
    # Initial Condition: Hot spot in the middle
    u = np.exp(-100 * ((X - L/2)**2 + (Y - W/2)**2))
    u_new = np.zeros_like(u)
    
    for n in range(Nt):
        # Fully Vectorized 5-point stencil update for interior points
        u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + \
            cfl_x * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) + \
            cfl_y * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])
        
        # Enforce Dirichlet Boundary Conditions (u=0 at edges)
        u_new[0, :] = 0; u_new[-1, :] = 0
        u_new[:, 0] = 0; u_new[:, -1] = 0
        
        u[:] = u_new[:]
        
    return X, Y, u

if __name__ == "__main__":
    X, Y, U = solve_heat_2d_explicit(Nx=40, Ny=40, Nt=1000, L=1.0, W=1.0, T=0.05, alpha=1.0)
    
    plt.figure(figsize=(6,5))
    plt.contourf(X, Y, U, 50, cmap='inferno')
    plt.colorbar(label='Temperature')
    plt.title('2D Heat Equation (Explicit Scheme)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('heat_result.png')
    print("Saved plot as heat_result.png")