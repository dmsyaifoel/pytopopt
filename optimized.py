'''
Highly optimized topopt python code using numpy magic. Comes at a significant cost to readability.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d
def oc(nelx, nely, x, volfrac, dcdx, l1=0, l2=1e5, ltol=1e-4, move=.2, xmin=1e-3, power=.5):
  while l2 - l1 > ltol: # this function is the optimiality criteria update step
    lmid = (l2 + l1)/2 # which has been extensively documented
    xnew = x*(dcdx/lmid)**power
    xnew = np.clip(xnew, x - move, x + move)
    xnew = np.clip(xnew, xmin, 1)
    if np.sum(xnew) > volfrac*nelx*nely: l1 = lmid
    else: l2 = lmid
  return xnew
def topopt(nelx, nely, fixeddofs, forces, volfrac, E=1, nu=.3, p=3, tol=.01, maxloops=100, r=1.5):
  ndof = 2*(nelx + 1)*(nely + 1)
  freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
  f = np.zeros(ndof)
  for force in forces: f[force[0]] = force[1]
  x = volfrac*np.ones((nely, nelx)) # the next two lines calculate the element stiffnes matrix
  k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8, -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
  Ke = np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]], [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]], [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]], [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]], [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]], [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]], [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]], [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])*E/(1 - nu**2)
  elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely)) # from here we set up dofs, which contains the dofs of all elements
  elx = elx.ravel()
  ely = ely.ravel()
  n1 = (nely + 1)*elx + ely
  n2 = (nely + 1)*(elx + 1) + ely
  dofs = np.array([2*n1, 2*n1 + 1, 2*n2, 2*n2 + 1, 2*n2 + 2, 2*n2 + 3, 2*n1 + 2, 2*n1 + 3]).T
  r_ = int(r) # from here we set up the filter kernel
  y_, x_ = np.ogrid[-r_:r_+1, -r_:r_+1]
  kernel = np.maximum(r - np.sqrt(x_**2 + y_**2), 0)
  d = convolve2d(np.ones((nely, nelx)), kernel, mode='same', boundary='symm') # we precompute the denominator of the filter once
  for i in range(maxloops):
    xf = x.ravel() # the next line builds the total stiffness matrix
    K = csc_array(((Ke[None, :, :]*(xf ** p)[:, None, None]).ravel(), (np.repeat(dofs, 8, axis=1).ravel(), np.tile(dofs, (1, 8)).ravel())), shape=(ndof, ndof))
    u = np.zeros(ndof)
    u[freedofs] = spsolve(K[freedofs, :][:, freedofs], f[freedofs]) # solve the fea
    uall = u[dofs]
    call = np.einsum('ni,ij,nj->n', uall, Ke, uall) # calculate the cost function on this and the next line
    c = np.sum(xf**p*call)
    dcdx = convolve2d(x*(p*xf**(p - 1)*call).reshape(nely, nelx), kernel, mode='same', boundary='symm')/x/d # calculate and filter the sensitivity
    xold = x.copy()
    x = oc(nelx, nely, x, volfrac, dcdx)
    dx = np.max(np.abs(x - xold))
    vol = np.sum(x)/(nelx*nely)
    print(f'{i = } {c = :3.3} {vol = :3.3} {dx = :3.3}')
    if dx < tol: break
  plt.imshow(-x, cmap='gray', interpolation='none')
  plt.axis('equal')
  plt.show()
nelx = 40
nely = 30
volfrac = .3
fixeddofs = np.union1d(np.arange(0, 2*(nely + 1), 2), [2*(nelx + 1)*(nely + 1) - 1])
forces = [[1, -1]]
x = topopt(nelx, nely, fixeddofs, forces, volfrac)
