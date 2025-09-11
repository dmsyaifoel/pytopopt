'''
Topology optimization for a 2 input 2 output compliant mechanism
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d
import random
def oc(nelx, nely, x, volfrac, dcdx, l1=0, l2=1e5, ltol=1e-4, move=.2, xmin=1e-3, power=.3, l2min=1e-40, dcdxmin=1e-10):
  while (l2 - l1)/(l1 + l2) > ltol and l2 > l2min:
    lmid = (l2 + l1)/2
    xnew = x*np.maximum(dcdxmin, -dcdx/lmid)**power
    xnew = np.clip(xnew, x - move, x + move)
    xnew = np.clip(xnew, xmin, 1)
    if np.sum(xnew) > volfrac*nelx*nely: l1 = lmid
    else: l2 = lmid
  return xnew
def topopt(nelx, nely, fixeddofs, indofs, outdofs, volfrac, tol, maxloops, r, kspring, p=3, E=1, nu=.3):
  ndof = 2*(nelx + 1)*(nely + 1)
  freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
  F = np.zeros((ndof, 4))
  F[indofs[0], 0] = 1
  F[outdofs[0], 2] = 1
  F[indofs[1], 1] = 1
  F[outdofs[1], 3] = 1
  x = volfrac*np.ones((nely, nelx))
  k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8, -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
  Ke = np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]], [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]], [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]], [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]], [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]], [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]], [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]], [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])*E/(1 - nu**2)
  elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely))
  elx = elx.ravel()
  ely = ely.ravel()
  n1 = (nely + 1)*elx + ely
  n2 = (nely + 1)*(elx + 1) + ely
  dofs = np.array([2*n1, 2*n1 + 1, 2*n2, 2*n2 + 1, 2*n2 + 2, 2*n2 + 3, 2*n1 + 2, 2*n1 + 3]).T
  r_ = int(r) + 1
  y_, x_ = np.ogrid[-r_:r_+1, -r_:r_+1]
  kernel = np.maximum(r - np.sqrt(x_**2 + y_**2), 0)
  d = np.sum(kernel)
  for i in range(maxloops):
    xf = x.ravel()
    K = csc_array(((Ke[None, :, :]*(xf ** p)[:, None, None]).ravel(), (np.repeat(dofs, 8, axis=1).ravel(), np.tile(dofs, (1, 8)).ravel())), shape=(ndof, ndof))
    K[indofs[0], indofs[0]] += kspring
    K[indofs[1], indofs[1]] += kspring
    K[outdofs[0], outdofs[0]] += kspring
    K[outdofs[1], outdofs[1]] += kspring
    U = np.zeros((ndof, 4))
    U[freedofs, :] = spsolve(K[freedofs, :][:, freedofs], F[freedofs, :])
    ce1 = np.einsum('ni,ij,nj->n', U[:, 0][dofs], Ke, U[:, 2][dofs])
    ce2 = np.einsum('ni,ij,nj->n', U[:, 1][dofs], Ke, U[:, 3][dofs])
    ce = ce1 + ce2
    dcdx = convolve2d((-p*xf**(p - 1)*ce).reshape(nely, nelx), kernel, mode='same', boundary='symm')/x/d
    xold = x.copy()
    x = oc(nelx, nely, x, volfrac, dcdx)
    dx = np.max(np.abs(x - xold))
    vol = np.sum(x)/(nelx*nely)
    c = U[outdofs[0], 0] + U[outdofs[1], 1]
    print(f'{i = } {c = :3.3} {vol = :3.3} {dx = :3.3}')
    if dx < tol: break
  plt.imshow(-x, cmap='gray', interpolation='none')
  plt.axis('equal')
  plt.show()
  return U
nelx = 40
nely = 30
volfrac = .2
fixeddofs = [0, 1, 2, 3, 4, 5, 6, 7, 54, 55, 56, 57, 58, 59, 60, 61]
indofs = [2480, 2481]
outdofs = [26, 27]
tol = .1
maxloops = 100
r = 1.5
kspring = .1
U = topopt(nelx, nely, fixeddofs, indofs, outdofs, volfrac, tol, maxloops, r, kspring)
