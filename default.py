'''
Python-only topology optimization with focus on readability and commenting. Pretty slow
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve

def element_stiffness_matrix(E, nu):
  '''
  Returns the 8x8 stiffness matrix of a single rectangular element
  given an Young's modulus and Poisson's ratio
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

def get_element_dofs(elx, ely, nely):
  '''
  Returns the degrees of freedom of the (corner) nodes of the element
  located at (elx, ely).

  If the grid looks like:

  (3,5)      |      (4,5)      |      (5,5)
             |                 |
  ------x=180,y=181-------x=222,y=223------
             |                 |
  (3,6)      |      (4,6)      |      (5,6)
             |                 |
  ------x=182,y=183-------x=224,y=225------
             |                 |
  (3,7)      |      (4,7)      |      (5,7)

  then get_element_dofs(4, 6, 20) returns [180, 181, 222, 223, 224, 225, 182, 183]
  '''

  n1 = (nely + 1)*elx + ely
  n2 = (nely + 1)*(elx + 1) + ely

  return np.array([
    2*n1,
    2*n1 + 1,
    2*n2,
    2*n2 + 1,
    2*n2 + 2,
    2*n2 + 3,
    2*n1 + 2,
    2*n1 + 3
  ])

def total_stiffness_matrix(K_elem, nelx, nely, x, p):
  '''
  Assembles the total stiffness matrix of the system
  given the current density distribution x.
  '''

  ndof = 2*(nelx + 1)*(nely + 1)

  # We will set up a sparse matrix in the form:
  # coo_array([list of i indices], [list of j indices], [list of values])
  # That is what we are setting up here
  iK = np.zeros(64*nelx*nely, dtype=int) # There are 8*8 values per element matrix
  jK = np.zeros(64*nelx*nely, dtype=int)
  sK = np.zeros(64*nelx*nely)

  count = 0
  for elx in range(nelx):
    for ely in range(nely):
      edofs = get_element_dofs(elx, ely, nely)
      for i in range(8):
        for j in range(8):
          iK[count] = edofs[i]
          jK[count] = edofs[j]
          sK[count] = K_elem[i, j]*x[ely, elx]**p # K*x**p is the SIMP model
          count += 1

  # Finally build the actual spare matrix and reshape to square
  K = coo_array((sK, (iK, jK)), shape=(ndof, ndof))

  return K.tocsc() # before returning, convert to a form that can be used in spsolve()

def smoothen_dcdx(nelx, nely, x, dcdx, r=1.5):
  '''
  Apply a smoothing filter over the dcdx matrix
  '''
  dcdxnew = np.zeros((nely, nelx))

  for i in range(nelx):
    for j in range(nely):
      total = 0

      for k in range(max(i - int(r) - 1, 0), min(i + int(r) + 1, nelx)):
        for l in range(max(j - int(r) - 1, 0), min(j + int(r) + 1, nely)):
          # fac is a linear, radially symmetric function that starts at 1 and becomes zero as the radius reaches r or more
          fac = max(r - np.sqrt((i - k)**2 + (j - l)**2), 0)

          # accumulate the values of the neighboring cells
          total += fac
          dcdxnew[j, i] += fac*x[l, k]*dcdx[l, k]

      # divide again by the multipliers (basically calculating a weighted average)
      dcdxnew[j, i] /= (x[j, i]*total)

  return dcdxnew

def optimality_criteria_update_x(nelx, nely, x, volfrac, dcdx, l1=0, l2=1e5, ltol=1e-4, move=.2, xmin=1e-3, power=.5):
  '''
  Update the density distribution x using the optimality criteria (OC) algorithm
  '''
  while l2 - l1 > ltol: # Keep going until the lagrangian multiplier l is within tolerances
    lmid = (l2 + l1)/2 # We use a bisection algorithm to find this l
    xnew = x*(dcdx/lmid)**power # Then update x using this formula derived from the lagrangian (to some damping factor called power)
    xnew = np.clip(xnew, x - move, x + move) # Ensure each density at most a value called move away from the old value (to increase stability)
    xnew = np.clip(xnew, xmin, 1) # Ensure each density is at least xmin (a very small nonzero value) and at most 1

    if np.sum(xnew) > volfrac*nelx*nely: # Check if the current value for l undershoots or overshoots the target volume fraction
      l1 = lmid # and use that to decide which half of the bisection algorithm to use next iteration
    else:
      l2 = lmid

  return xnew

def plot(x):
  '''
  Plot the values of x as a grayscale image
  '''
  plt.imshow(-x, cmap='gray', interpolation='none')
  plt.axis('equal')
  plt.axis('off')
  plt.show()

def topopt(nelx, nely, fixed_dofs, forces, volfrac, E=1, nu=.3, p=3, tol=.01, maxloops=100):
  '''
  Perform minimum compliance topology optimization on a (nelx, nely) grid given
  - a list of fixed dof numbers
  - a list of (dof, magnitude) force tuples
  - a target volume fraction

  - some parameters with default values
    - Young's modulus
    - Poisson's ratio
    - SIMP penalization factor
    - tolerance (stopping criterion)
    - maximum number of iteration loops
  '''
  ndofs = 2*(nelx + 1)*(nely + 1)
  free_dofs = np.setdiff1d(np.arange(ndofs), fixed_dofs) # free dofs are the difference between all dofs and the fixed dofs

  # setup the force vector f
  f = np.zeros(ndofs)
  for force in forces:
    f[force[0]] = force[1]

  x = volfrac*np.ones((nely, nelx)) # initialize density distribution with all cells at the target volume fraction

  K_elem = element_stiffness_matrix(E, nu)

  for i in range(maxloops):
    K = total_stiffness_matrix(K_elem, nelx, nely, x, p) # Get the current stiffness matrix given the current density distribution

    u = np.zeros(ndofs)
    u[free_dofs] = spsolve(K[free_dofs, :][:, free_dofs], f[free_dofs]) # solve K@u - f for the displacement given the current density distribution
    # These slices (u[], K[][], f[]) ensure we only use the dofs that are not fixed

    # Initialize the cost and sensitivity
    c = 0
    dcdx = np.zeros((nely, nelx))

    for elx in range(nelx):
      for ely in range(nely):
        edofs = get_element_dofs(elx, ely, nely)
        u_elem = u[edofs]
        c_elem = u_elem.dot(K_elem.dot(u_elem)) # for each element calculate the compliance u.T@K@u
        x_elem = x[ely, elx]
        c += x_elem**p*c_elem # add to the cost the compliance, multiplied by the density to the power of p (this is SIMP)
        dcdx[ely, elx] = p*x_elem**(p - 1)*c_elem # add the derivative of this cost to the senstivity

    dcdx = smoothen_dcdx(nelx, nely, x, dcdx) # smoothen the sensitivity matrix to avoid things like checkerboarding

    xold = x.copy()
    x = optimality_criteria_update_x(nelx, nely, x, volfrac, dcdx) # update the density distribution using the OC algorithm

    dx = np.max(np.abs(x - xold)) # calculate the largest change in value over all cells
    vol = np.sum(x)/(nelx*nely)
    print(f'{i = } {c = :3.3} {vol = :3.3} {dx = :3.3}')

    if dx < tol: # stop if the largest change in value over all cells is less than the tolerance
      break

  plot(x)
  return x

nelx = 40
nely = 30
volfrac = .3


fixed_dofs = np.union1d(
  np.arange(0, 2*(nely + 1), 2), # in this example we fix all nodes at the left border in x direction
  [2*(nelx + 1)*(nely + 1) - 1], # and the bottom right node in y direciton
)

forces = [[1, -1]] # and we add one vertical force in the top left node
# in other words, we design half of a bridge-like structure loaded in the middle and roller supported at the ends using mirror symmetry

x = topopt(nelx, nely, fixed_dofs, forces, volfrac)
