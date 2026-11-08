import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def solve_poisson_2d(Nx, Ny, L, W):
    """
    Solves the 2D Poisson equation: Laplace(u) = f
    Using a 5-point finite difference stencil and sparse matrices.
    """
    dx = L / (Nx + 1)
    dy = W / (Ny + 1)
    
    # 1. Create 1D discrete Laplacian matrices for x and y directions
    # Diagonals: [1, -2, 1] / dx^2
    D_x = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)) / (dx**2)
    D_y = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny)) / (dy**2)
    
    # 2. Construct the 2D Laplacian using Kronecker Sum
    # L_2D = I_y (kron) D_x + D_y (kron) I_x
    L_2D = sp.kronsum(D_y, D_x, format='csc')
    
    # 3. Define the right-hand side function f(x, y)
    # Example: f(x,y) = -2 * pi^2 * sin(pi*x) * sin(pi*y)
    x = np.linspace(dx, L-dx, Nx)
    y = np.linspace(dy, W-dy, Ny)
    X, Y = np.meshgrid(x, y)
    
    # Flatten the 2D grid into a 1D vector for the linear solver
    f = (-2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)).flatten()
    
    # 4. Solve the linear system L_2D * u = f
    u_vec = spla.spsolve(L_2D, f)
    
    # Reshape back to 2D grid
    u_2d = u_vec.reshape((Ny, Nx))
    
    return X, Y, u_2d

if __name__ == "__main__":
    X, Y, U = solve_poisson_2d(Nx=50, Ny=50, L=1.0, W=1.0)
    
    plt.figure(figsize=(6,5))
    plt.contourf(X, Y, U, 50, cmap='viridis')
    plt.colorbar(label='Solution u(x,y)')
    plt.title('2D Poisson Equation Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('poisson_result.png')
    print("Saved plot as poisson_result.png")