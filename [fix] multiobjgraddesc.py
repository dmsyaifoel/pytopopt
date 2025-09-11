

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d

scale = 1
nx, ny = scale*40, scale*30
nu = .3
E = 1
r = 1.5
low = 60
high = 10*low
loops = 50
xmin = .01
p = 3
vf = .2
move = .2

def get_element_dofs(elx, ely, nely):
  n1 = (nely + 1)*elx + ely
  return 2*n1, 2*n1 + 1

fixeddofs = []
for i in range(ny//12):
  fixeddofs.append(get_element_dofs(elx=0, ely=i, nely=ny)[0])
  fixeddofs.append(get_element_dofs(elx=0, ely=i, nely=ny)[1])
for i in range(11*ny//12, ny):
  fixeddofs.append(get_element_dofs(elx=0, ely=i, nely=ny)[0])
  fixeddofs.append(get_element_dofs(elx=0, ely=i, nely=ny)[1])
fixeddofs = np.array(fixeddofs)
print(fixeddofs)

indofs = [
  get_element_dofs(elx=0, ely=2*ny//6, nely=ny)[0],
  get_element_dofs(elx=0, ely=2*ny//6, nely=ny)[1],
  get_element_dofs(elx=0, ely=3*ny//6, nely=ny)[0],
  get_element_dofs(elx=0, ely=3*ny//6, nely=ny)[1],
  get_element_dofs(elx=0, ely=4*ny//6, nely=ny)[0],
  get_element_dofs(elx=0, ely=4*ny//6, nely=ny)[1],
]
indofs=np.array(indofs)
print(indofs)

outdofs = [
  get_element_dofs(elx=nx//3, ely=0, nely=ny)[1],
  get_element_dofs(elx=nx//3, ely=0, nely=ny)[0],
  get_element_dofs(elx=2*nx//3, ely=0, nely=ny)[1],
  get_element_dofs(elx=2*nx//3, ely=0, nely=ny)[0],
  get_element_dofs(elx=nx, ely=0, nely=ny)[1],
  get_element_dofs(elx=nx, ely=0, nely=ny)[0],
]
outdofs=np.array(outdofs)
print(outdofs)

ndof = 2*(nx + 1)*(ny + 1)
freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

pairs = len(indofs)
assert pairs == len(outdofs)
cases = 3*pairs

in_idx  = np.abs(indofs)
out_idx = np.abs(outdofs)
in_sgn  = np.sign(indofs)
out_sgn = np.sign(outdofs)

cols_in   = np.arange(pairs)
cols_out  = np.arange(pairs, 2*pairs)
cols_both = np.arange(2*pairs, 3*pairs)

rows = np.concatenate([in_idx, out_idx, in_idx, out_idx])
cols = np.concatenate([cols_in, cols_out, cols_both, cols_both])
data = np.concatenate([in_sgn, out_sgn, in_sgn, out_sgn])

F = csc_array((data, (rows, cols)), shape=(ndof, cases))

F = F.todense()
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

x = np.ones((ny, nx))

def oc(nx, ny, x, volfrac, dcdx, l1=0, l2=1e5, ltol=1e-4, move=.2, xmin=1e-3, power=.5):
  while l2 - l1 > ltol:
    lmid = (l2 + l1)/2
    xnew = x*(np.abs(dcdx)/lmid)**power
    xnew = np.clip(xnew, x - move, x + move)
    xnew = np.clip(xnew, xmin, 1)
    if np.sum(xnew) > volfrac*nx*ny: l1 = lmid
    else: l2 = lmid
  return xnew

try:
  for i in range(loops):
    dc_ = np.zeros_like(x)

    K = csc_array(((Ke[None, :, :]*(x.ravel()**p)[:, None, None]).ravel(), (np.repeat(dofs, 8, axis=1).ravel(), np.tile(dofs, (1, 8)).ravel())), shape=(ndof, ndof))
    U = np.zeros_like(F)
    U[freedofs] = spsolve(K[freedofs, :][:, freedofs], F[freedofs])

    h = []
    hh = []
    for j in range(cases):
      Uall = U[dofs, j]
      call = np.einsum('ni,ij,nj->n', Uall, Ke, Uall).reshape((ny, nx))

      if j < 2*pairs: c = np.sum(x**p*call) - low
      else: c = high - np.sum(x**p*call)

      h.append(int(np.sum(x**p*call)))
      hh.append(str(int(c < 0)))

      if j < 2*pairs: dc = p*x**(p - 1)*call
      else: dc = -p*x**(p - 1)*call *1000

      dc = convolve2d(x*dc, kernel, mode='same', boundary='symm')/x/d
      dc_ += dc

    x = oc(nx, ny, x, vf, dc_, power=.3)
    print(f'{i} {"".join(hh)} {np.min(x):3.3} {np.max(x):3.3} {np.sum(x)/nx/ny:3.3} {h}')

except:
  pass

plt.imshow(-x, cmap='gray', interpolation='none')
plt.axis('equal')
plt.show()