import jax.numpy as jnp
import numpy as np
import pymoto as pym
from datetime import datetime
import shutil


def split(A): return A[:6, :6], A[6:, :6] #, A[:6, 6:], A[6:, 6:] # comment out last two terms to ignore effect of forces on output

def cat(a, b): return jnp.concatenate([a, b])
# def cat(a, b, c, d): return jnp.concatenate([a, b, c, d])

timestamp = datetime.now().strftime('%y%m%d-%H%M%S')

scale = 5
nx = scale*40
ny = scale*30
vf = .5
xmin = 1e-3

domain = pym.VoxelDomain(nx, ny)
nodes_left = domain.nodes[0, :].ravel()
fixed_nodes = np.concatenate([nodes_left[:ny//6], nodes_left[5*ny//6:]])
fixed_dofs = domain.get_dofnumber(fixed_nodes).ravel()
output_nodes = np.array([nodes_left[2*ny//6], nodes_left[3*ny//6], nodes_left[4*ny//6]])
output_dofs = domain.get_dofnumber(output_nodes).ravel()
nodes_top = domain.nodes[:, 0].ravel()
input_nodes = np.array([nodes_top[1*nx//3], nodes_top[2*nx//3], nodes_top[nx]])
input_dofs = domain.get_dofnumber(input_nodes).ravel()
all_dofs = np.arange(0, 2*domain.nnodes)
main_dofs = np.concatenate([input_dofs, output_dofs])
free_dofs = np.setdiff1d(all_dofs, np.concatenate([fixed_dofs, main_dofs]))

x = pym.Signal('x', state=vf*np.ones(domain.nel))

with pym.Network() as fn:
  x_filtered = pym.DensityFilter(domain, radius=2)(x)
  x_filtered.tag = 'Density'

  vol = pym.EinSum('i->')(x)

  pym.PlotDomain(domain, saveto = timestamp + '/')(x_filtered)

  x_simp = pym.MathExpression(f'{xmin} + {1 - xmin}*inp0')(x_filtered)

  K = pym.AssembleStiffness(domain)(x_simp)

  K_condensed = pym.StaticCondensation(main=main_dofs, free=free_dofs)(K)

  C = pym.Inverse()(K_condensed)
  C2 = pym.EinSum('ij,ij->ij')(C, C)

  Ctot = pym.EinSum('ij->')(C2)


  quadrants = pym.AutoMod(split)(C2)

  A = np.eye(6)

  diags = []
  smallestdiags = []
  traces = []
  sums = []
  offsums = []
  diagmins = []
  divs = []

  for i, quadrant in enumerate(quadrants):
    diags.append(pym.EinSum('ij,ij->i')(A, quadrant)) # vectors containing the diagonals
    smallestdiags.append(pym.SoftMinMax(alpha=-1)(diags[-1])) # smallest of the diagonals (per quadrant)
    traces.append(pym.EinSum('i->')(diags[-1])) # trace, scalar, sum of diags
    sums.append(pym.EinSum('ij->')(quadrant)) # total sum of the quadrant
    offsums.append(pym.MathExpression('inp0 - inp1')(sums[-1], traces[-1])) # sum of the off-diagonals
    divs.append(pym.MathExpression('inp0/inp1')(traces[-1], offsums[-1])) # division, might be useful

  alldiags = pym.AutoMod(cat)(*diags) # vector of all diagonals

  smallestofalldiags = pym.SoftMinMax(alpha=-1)(alldiags) # smallest value of all diagonals

  add = '1' + ''.join([f'+inp{i}' for i in range(len(quadrants))])
  mul = '1' + ''.join([f'*inp{i}' for i in range(len(quadrants))])

  alltraces = pym.MathExpression(add)(*traces)
  alloffsums = pym.MathExpression(add)(*offsums)

  # obj = pym.MathExpression('inp0/inp1*(inp2+1)')(alloffsums, smallestofalldiags, vol)
  obj = pym.MathExpression('inp0 - inp1')(alloffsums, traces[-1])

  obj_con = [obj]

  pym.PlotIter()(*obj_con)

shutil.copy2(__file__, timestamp + '/moto.py')

pym.minimize_mma(x, obj_con)
