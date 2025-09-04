import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d
from random import random
def topopt(nx, ny, ctarget, volfrac, fixeddofs, forces, maxloops, r=1.5, p=3, E=1, nu=.3, xmin=1e-3):
  ndof = 2*(nx + 1)*(ny + 1)
  freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
  f = np.zeros(ndof)
  for force in forces: f[force[0]] = force[1]
  k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8, -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
  Ke = np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]], [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]], [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]], [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]], [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]], [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]], [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]], [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])*E/(1 - nu**2)
  _x, _y = np.meshgrid(np.arange(nx), np.arange(ny))
  _x = _x.ravel()
  _y = _y.ravel()
  n1 = (ny + 1)*_x + _y
  n2 = (ny + 1)*(_x + 1) + _y
  dofs = np.array([2*n1, 2*n1 + 1, 2*n2, 2*n2 + 1, 2*n2 + 2, 2*n2 + 3, 2*n1 + 2, 2*n1 + 3]).T
  r_ = int(r) + 1
  y_, x_ = np.ogrid[-r_:r_+1, -r_:r_+1]
  kernel = np.maximum(r - np.sqrt(x_**2 + y_**2), 0)
  d = np.sum(kernel)
  dg = np.ones((ny, nx))
  x = np.ones((ny, nx))
  dg_ = dg/np.sum(dg*dg)
  for i in range(maxloops):
    K = csc_array(((Ke[None, :, :]*(x.ravel()**p)[:, None, None]).ravel(), (np.repeat(dofs, 8, axis=1).ravel(), np.tile(dofs, (1, 8)).ravel())), shape=(ndof, ndof))
    u = np.zeros(ndof)
    u[freedofs] = spsolve(K[freedofs, :][:, freedofs], f[freedofs])
    uall = u[dofs]
    call = np.einsum('ni,ij,nj->n', uall, Ke, uall).reshape((ny, nx))
    c = np.sum(x**p*call) - ctarget
    dx = np.zeros_like(x)
    if c > 0:
      dc = p*x**(p - 1)*call
      dc = convolve2d(x*dc, kernel, mode='same', boundary='symm')/x/d
      dx += dc/np.sum(dc*dc)*c
    g = np.sum(x) - volfrac*nx*ny
    if g > 0:
      dx -= dg_*g
    x += dx
    x = np.clip(x, xmin, 1)
    print(f'{i} {np.sum(x**p*call):3.3} {np.sum(x)/nx/ny:3.3}')
  plt.imshow(-x, cmap='gray', interpolation='none')
  plt.axis('equal')
  plt.show()
  return x
nx, ny = 80, 60
fixeddofs = np.union1d(np.arange(0, 2*(ny + 1), 2), np.array([2*(nx + 1)*(ny + 1) - 1]))
x = topopt(nx, ny, 50, .2, fixeddofs, [[1, -1]], 50)
