'''
Highly optimized topopt python code using numpy magic. Comes at a significant cost to readability.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d # Import for vectorized smoothing

def element_stiffness_matrix(E, nu):
    '''
    Returns the 8x8 stiffness matrix of a single rectangular element
    given an Young's modulus and Poisson's ratio.
    This function is already vectorized and efficient.
    '''

    k = np.array([
        1/2 - nu/6,
        1/8 + nu/8,
        -1/4 - nu/12,
        -1/8 + 3*nu/8,
        -1/4 + nu/12,
        -1/8 - nu/8,
        nu/6,
        1/8 - 3*nu/8
    ])

    M = np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])

    return M*E/(1 - nu**2)

def get_all_element_dofs(nelx, nely):
    '''
    Returns the degrees of freedom (DOFs) for all elements in a vectorized manner.
    This replaces the original `get_element_dofs` function for overall efficiency.
    '''
    # Create meshgrid for all element x and y coordinates
    elx_indices, ely_indices = np.meshgrid(np.arange(nelx), np.arange(nely))

    # Flatten these to get a list of (elx, ely) pairs
    elx_flat = elx_indices.flatten()
    ely_flat = ely_indices.flatten()

    # Calculate node indices for the bottom-left (n1) and bottom-right (n2) nodes of each element
    n1_all = (nely + 1) * elx_flat + ely_flat
    n2_all = (nely + 1) * (elx_flat + 1) + ely_flat

    # Construct the 8 DOFs for each element.
    # Each row in `edofs_all` corresponds to an element, and its 8 columns
    # represent the DOFs of its four corner nodes (x and y displacement for each).
    # The order of DOFs here must match the order expected by `element_stiffness_matrix`.
    edofs_all = np.array([
        2*n1_all,       # x-DOF of bottom-left node
        2*n1_all + 1,   # y-DOF of bottom-left node
        2*n2_all,       # x-DOF of bottom-right node
        2*n2_all + 1,   # y-DOF of bottom-right node
        2*n2_all + 2,   # x-DOF of top-right node (this is n2+1, since n2 is bottom-right)
        2*n2_all + 3,   # y-DOF of top-right node
        2*n1_all + 2,   # x-DOF of top-left node (this is n1+1, since n1 is bottom-left)
        2*n1_all + 3    # y-DOF of top-left node
    ]).T # Transpose to get shape (num_elements, 8)

    return edofs_all

def total_stiffness_matrix(K_elem, nelx, nely, x, p, edofs_all):
    '''
    Assembles the total stiffness matrix of the system given the current
    density distribution x, using vectorized operations.
    '''
    ndof = 2*(nelx + 1)*(nely + 1)

    # Prepare indices for the sparse matrix (iK for rows, jK for columns)
    # Each element's 8x8 stiffness matrix contributes to the global matrix.
    # We need to map each entry of the 8x8 K_elem to its global (i, j) index.
    # iK: Repeat each element's DOFs 8 times (for the 8 columns of K_elem)
    iK = np.repeat(edofs_all, 8, axis=1).flatten()
    # jK: Tile each element's DOFs 8 times (for the 8 rows of K_elem)
    jK = np.tile(edofs_all, (1, 8)).flatten()

    # Calculate sK values: K_elem * x_element**p
    # x is (nely, nelx), flatten it to match the element order (row by row)
    x_flat = x.flatten()
    # Broadcast K_elem to each element and multiply by penalized density
    sK_values_per_element = K_elem[np.newaxis, :, :] * (x_flat**p)[:, np.newaxis, np.newaxis]
    sK = sK_values_per_element.flatten()

    # Finally build the actual sparse matrix using COO format
    K = coo_array((sK, (iK, jK)), shape=(ndof, ndof))

    # Convert to CSC (Compressed Sparse Column) format for efficient column slicing,
    # which is used by `spsolve`.
    return K.tocsc()

def make_filter_kernel(r):
    '''
    Creates a circular filter kernel based on the radius r.
    This kernel is used for smoothing the sensitivity.
    '''
    r_int = int(r)
    size = 2 * r_int + 1 # Kernel size (e.g., r=1.5 -> r_int=1 -> size=3)
    kernel = np.zeros((size, size))
    center = r_int # Center index of the kernel

    # Populate the kernel with weights based on radial distance
    for dx in range(-r_int, r_int + 1):
        for dy in range(-r_int, r_int + 1):
            dist = np.sqrt(dx**2 + dy**2)
            kernel[center + dy, center + dx] = max(r - dist, 0)
    return kernel

def smoothen_dcdx(nelx, nely, x, dcdx, r=1.5):
    '''
    Apply a smoothing filter over the dcdx matrix using vectorized convolution.
    This helps prevent checkerboarding and ensures a smoother design.
    '''
    kernel = make_filter_kernel(r)

    # Numerator of the smoothed sensitivity formula: sum(fac * x_neighbor * dcdx_neighbor)
    # This is equivalent to convolving (x * dcdx) with the filter kernel.
    numerator = convolve2d(x * dcdx, kernel, mode='same', boundary='symm')

    # Denominator of the smoothed sensitivity formula: x_current * sum(fac)
    # sum(fac) for each cell's neighborhood is obtained by convolving a matrix of ones with the kernel.
    denominator_sum_fac = convolve2d(np.ones((nely, nelx)), kernel, mode='same', boundary='symm')
    denominator = x * denominator_sum_fac

    # Calculate the new smoothed sensitivity, adding a small epsilon to prevent division by zero
    dcdxnew = numerator / (denominator + 1e-9)

    return dcdxnew

def optimality_criteria_update_x(nelx, nely, x, volfrac, dcdx, l1=0, l2=1e5, ltol=1e-4, move=.2, xmin=1e-3, power=.5):
    '''
    Update the density distribution x using the optimality criteria (OC) algorithm.
    This is an iterative bisection method to find the Lagrange multiplier 'lmid'
    that satisfies the volume constraint.
    '''
    while l2 - l1 > ltol: # Keep iterating until the Lagrangian multiplier 'lmid' converges
        lmid = (l2 + l1)/2 # Bisection step
        # Update x using the OC formula, damped by 'power'
        xnew = x*(dcdx/lmid)**power
        # Apply 'move' limit to ensure stability (densities don't change too drastically)
        xnew = np.clip(xnew, x - move, x + move)
        # Ensure densities stay within [xmin, 1] bounds
        xnew = np.clip(xnew, xmin, 1)

        # Check if the current volume fraction is above or below the target
        if np.sum(xnew) > volfrac*nelx*nely:
            l1 = lmid # If volume is too high, increase lmid (makes densities lower)
        else:
            l2 = lmid # If volume is too low, decrease lmid (makes densities higher)

    return xnew

def plot(x):
    '''
    Plot the values of x (density distribution) as a grayscale image.
    Black represents solid material (density 1), white represents void (density 0).
    '''
    plt.imshow(-x, cmap='gray', interpolation='none')
    plt.axis('equal')
    plt.axis('off')
    plt.show()

def topopt(nelx, nely, fixed_dofs, forces, volfrac, plot_every=5, E=1, nu=.3, p=3, tol=.01, maxloops=100):
    '''
    Perform minimum compliance topology optimization on a (nelx, nely) grid.

    Parameters:
    - nelx, nely: Number of elements in x and y directions.
    - fixed_dofs: List of global degrees of freedom (DOFs) that are fixed (boundary conditions).
    - forces: List of (dof, magnitude) tuples representing applied forces.
    - volfrac: Target volume fraction (percentage of material in the design domain).
    - plot_every: How often to plot the design (in iterations).
    - E: Young's modulus of the material.
    - nu: Poisson's ratio of the material.
    - p: SIMP penalization factor (controls how intermediate densities are penalized).
    - tol: Tolerance for convergence (stopping criterion based on change in density).
    - maxloops: Maximum number of optimization iterations.
    '''
    ndofs = 2*(nelx + 1)*(nely + 1) # Total number of degrees of freedom
    # Determine free DOFs by removing fixed DOFs from all possible DOFs
    free_dofs = np.setdiff1d(np.arange(ndofs), fixed_dofs)

    # Setup the global force vector f
    f = np.zeros(ndofs)
    for force in forces:
        f[force[0]] = force[1] # Apply force at specified DOF

    # Initialize density distribution with all elements at the target volume fraction
    x = volfrac*np.ones((nely, nelx))

    # Calculate the stiffness matrix for a single element once
    K_elem = element_stiffness_matrix(E, nu)

    # Pre-calculate all element DOFs once, as they don't change during optimization
    edofs_all = get_all_element_dofs(nelx, nely)

    # Main optimization loop
    for i in range(maxloops):
        # 1. FEA: Get the current global stiffness matrix and solve for displacements
        K = total_stiffness_matrix(K_elem, nelx, nely, x, p, edofs_all)
        u = np.zeros(ndofs)
        # Solve for displacements only for the free DOFs
        u[free_dofs] = spsolve(K[free_dofs, :][:, free_dofs], f[free_dofs])

        # 2. Compliance and Sensitivity Calculation (Vectorized)
        # Extract displacements for all elements based on pre-calculated DOFs
        u_all_elements = u[edofs_all] # Shape (num_elements, 8)

        # Calculate element compliance (u_elem.T @ K_elem @ u_elem) for all elements
        # Using np.einsum for efficient batch matrix multiplication
        # 'ni,ij,nj->n' means: for each element 'n', sum over 'i' and 'j' (element DOFs)
        # (u_n_i * K_i_j * u_n_j) to get a scalar for each element 'n'.
        c_elem_all = np.einsum('ni,ij,nj->n', u_all_elements, K_elem, u_all_elements)

        # Flatten x for element-wise operations
        x_flat = x.flatten()

        # Total compliance (cost function)
        c = np.sum(x_flat**p * c_elem_all)

        # Sensitivity (derivative of cost with respect to density x)
        # dC/dx_e = p * x_e^(p-1) * (u_e.T @ K_elem @ u_e)
        dcdx_flat = p * x_flat**(p - 1) * c_elem_all
        dcdx = dcdx_flat.reshape(nely, nelx) # Reshape back to grid form

        # 3. Filter the sensitivity to avoid numerical instabilities (e.g., checkerboarding)
        dcdx = smoothen_dcdx(nelx, nely, x, dcdx)

        # Store old density for convergence check
        xold = x.copy()
        # 4. Update densities using the Optimality Criteria algorithm
        x = optimality_criteria_update_x(nelx, nely, x, volfrac, dcdx)

        # 5. Check for convergence and print status
        dx = np.max(np.abs(x - xold)) # Largest change in density
        vol = np.sum(x)/(nelx*nely) # Current volume fraction
        print(f'{i = } {c = :3.3} {vol = :3.3} {dx = :3.3}')

        # Plot the current design periodically
        if i % plot_every == 0:
            plot(x)

        # Stop if the change in density is below tolerance
        if dx < tol:
            print(f"Converged after {i+1} iterations.")
            break
    else: # This block executes if the loop completes without 'break' (i.e., maxloops reached)
        print(f"Maximum iterations ({maxloops}) reached.")

    # Plot the final optimized design
    plot(x)
    return x

# --- Example Usage (from original notebook) ---
nelx = 200
nely = 100
volfrac = .3

# Define fixed degrees of freedom (boundary conditions)
# Fix all nodes at the left border in x direction
# And fix the bottom-right node in y direction
fixed_dofs = np.union1d(
    np.arange(0, 2*(nely + 1), 2), # x-DOFs at x=0 (left edge)
    [2*(nelx + 1)*(nely + 1) - 1], # y-DOF of the very last node (bottom-right)
)

# Define applied forces
# Apply one vertical force (magnitude -1, i.e., downwards) at the top-left node (DOF 1)
# (Node 0 is at (0,0), its x-DOF is 0, y-DOF is 1)
forces = [[1, -1]]

# Run the topology optimization
x = topopt(nelx, nely, fixed_dofs, forces, volfrac)
