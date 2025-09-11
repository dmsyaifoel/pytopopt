import numpy as np # the same code as default.py optimized for speed and line count. Completely unreadable
import matplotlib.pyplot as plt
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d
def oc(nx, ny, x, volfrac, dcdx, l1=0, l2=1e5, ltol=1e-4, move=.2, xmin=1e-3, power=.5):
  while l2 - l1 > ltol:
    lmid = (l2 + l1)/2
    xnew = x*(dcdx/lmid)**power
    xnew = np.clip(xnew, x - move, x + move)
    xnew = np.clip(xnew, xmin, 1)
    if np.sum(xnew) > volfrac*nx*ny: l1 = lmid
    else: l2 = lmid
  return xnew
def topopt(nx, ny, volfrac, fixeddofs, forces, tol, maxloops, r, p=3, E=1, nu=.3):
  ndof = 2*(nx + 1)*(ny + 1)
  freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
  f = np.zeros(ndof)
  for force in forces: f[force[0]] = force[1]
  x = volfrac*np.ones((ny, nx))
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
  for i in range(maxloops):
    xf = x.ravel()
    K = csc_array(((Ke[None, :, :]*(xf ** p)[:, None, None]).ravel(), (np.repeat(dofs, 8, axis=1).ravel(), np.tile(dofs, (1, 8)).ravel())), shape=(ndof, ndof))
    u = np.zeros(ndof)
    u[freedofs] = spsolve(K[freedofs, :][:, freedofs], f[freedofs])
    uall = u[dofs]
    call = np.einsum('ni,ij,nj->n', uall, Ke, uall)
    c = np.sum(xf**p*call)
    dcdx = convolve2d(x*(p*xf**(p - 1)*call).reshape(ny, nx), kernel, mode='same', boundary='symm')/x/d # 
    xold = x.copy()
    x = oc(nx, ny, x, volfrac, dcdx)
    dx = np.max(np.abs(x - xold))
    vol = np.sum(x)/(nx*ny)
    print(f'{i = } {c = :3.3} {vol = :3.3} {dx = :3.3}')
    if dx < tol: break
  plt.imshow(-x, cmap='gray', interpolation='none')
  plt.axis('equal')
  plt.show()
  return x
nx, ny = 40, 30
fixeddofs = np.union1d(np.arange(0, 2*(ny + 1), 2), [2*(nx + 1)*(ny + 1) - 1])
x = topopt(nx, ny, .3, fixeddofs, [[1, -1]], .01, 1000, 1.5)