import jax.numpy as jnp
import numpy as np
import pymoto as pym
from datetime import datetime
import shutil


def split(A): return A[:6, :6], A[6:, :6], A[:6, 6:], A[6:, 6:] # comment out last two terms to ignore effect of forces on output

# def cat(a, b): return jnp.concatenate([a, b])
def cat(a, b, c, d): return jnp.concatenate([a, b, c, d])

timestamp = datetime.now().strftime('%Y%m%d-%H%M')

scale = 5
nx = scale*40
ny = scale*30
vf = .2
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
  volcon = pym.MathExpression(f'inp0 - {vf*nx*ny}')(vol)

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
  diagsedit = []
  diagsedittrace = []
  smallestdiags = []
  traces = []
  sums = []
  offsums = []
  diagmins = []
  divs = []

  diags = [pym.EinSum('ij,ij->i')(A, quadrant)) for quadrant in quadrants] # vectors containing the diagonals
  traces = [pym.EinSum('ij,ij->')(A, quadrant)) for quadrant in quadrants] # traces (scalars, sums of diags)
  sums = [pym.EinSum('ij->')(A, quadrant)) for quadrant in quadrants]
  offsums = [pym.MathExpression('inp0 - inp1')(sums[i], traces[i]) for i in range(len(quadrants))]

  alldiags = pym.AutoMod(cat)(*diags) # vector of all diagonals

  # smallestalldiags = pym.SoftMinMax(alpha=-10)(alldiags) # smallest value of all diagonals

  add = ''.join([f'+inp{i}' for i in range(len(quadrants))])
  mul = '1' + ''.join([f'*inp{i}' for i in range(len(quadrants))])

  alltraces = pym.MathExpression(add)(*traces)
  alloffsums = pym.MathExpression(add)(*offsums)
  alldivs = pym.MathExpression(add)(*divs)
  alldet = pym.MathExpression(add)(*diagsedittrace)

  obj = alldivs

  # obj = pym.MathExpression('inp0/inp1')(alloffsums, alltraces)

  # obj = pym.MathExpression('inp0/inp1*(inp2+1)')(alloffsums, smallestofalldiags, vol)
  # obj = pym.MathExpression('inp0/inp1')(alloffsums, smallestalldiags)

  # aoscon = pym.MathExpression('inp0 - 1e-3')(alloffsums)
  # sd0con = pym.MathExpression('-inp0 + 2000')(smallestdiags[0])
  # sd1con = pym.MathExpression('-inp0 + 5000')(smallestdiags[1])

  # obj_con = [vol, sd0con, sd1con]
  obj_con = [obj]

  # obj_con = [alloffsums, volcon, pym.MathExpression('-inp0 + 500')(smallestdiags[-1])]


  pym.PlotIter()(*obj_con)

shutil.copy2(__file__, timestamp + '/moto.py')

pym.minimize_mma(x, obj_con, maxit=1000)
