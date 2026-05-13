import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve
import matplotlib
from matplotlib.pyplot import imshow, show
nu, E = .3, 1
k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8, -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
Ke = np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],])*E/(1 - nu**2)
Ke_ = np.array([Ke])
nx, ny = 40, 30
nelem = nx*ny
ndof = 2*(nx + 1)*(ny + 1)
u = np.zeros(ndof)
fixeddofs = np.union1d(np.arange(0, 2*(ny + 1), 2), np.array([2*(nx + 1)*(ny + 1) - 1]))
freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
f = np.zeros(ndof)
f[1] = -1
_x, _y = np.meshgrid(np.arange(nx), np.arange(ny))
_x, _y = _x.ravel(), _y.ravel()
n1, n2 = (ny + 1)*_x + _y, (ny + 1)*(_x + 1) + _y
dofs = np.array([2*n1, 2*n1 + 1, 2*n2, 2*n2 + 1, 2*n2 + 2, 2*n2 + 3, 2*n1 + 2, 2*n1 + 3]).T
rows = np.repeat(dofs, 8, axis=1).ravel()
cols = np.tile(dofs, (1, 8)).ravel()
vf, p, lr = .3, 3, .05
x = np.ones(nelem)*vf
for i in range(100):
  K = csc_array(((Ke*(x**p)[:, None, None]).ravel(), (rows, cols)), shape=(ndof, ndof))
  u[freedofs] = spsolve(K[freedofs, :][:, freedofs], f[freedofs])
  uall = u[dofs]
  call = np.einsum('ni,ij,nj->n', uall, Ke, uall)
  c = np.sum(call)
  dc = -p*x**(p - 1)*call
  vol = np.average(x)
  print(f'{i = }, {c = }, {vol = }')
  x -= dc*lr
  if vol > vf: x -= vol - vf
  x = np.clip(x, 1e-3, 1)
imshow(-x.reshape((ny, nx)), cmap='gray')
show()
