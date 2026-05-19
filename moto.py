import numpy as np
from datetime import datetime
import shutil
from matplotlib.pyplot import imshow, show
import pymoto as pym

timestamp = datetime.now().strftime('%Y%m%d-%H%M')

scale = 4
nx = scale*40
ny = scale*30

vf=.5

xmin = 1e-3

domain = pym.VoxelDomain(nx, ny)

nodes_left = domain.nodes[0, :].ravel()
fixed_nodes = np.concatenate([nodes_left[:ny//6], nodes_left[5*ny//6:]])
fixed_dofs = domain.get_dofnumber(fixed_nodes).ravel()

output_nodes = np.array([nodes_left[2*ny//6], nodes_left[3*ny//6], nodes_left[4*ny//6]])
# output_nodes = np.array(nodes_left[3*ny//6])

output_dofs = domain.get_dofnumber(output_nodes).ravel()

nodes_top = domain.nodes[:, 0].ravel()
input_nodes = np.array([nodes_top[1*nx//3], nodes_top[2*nx//3], nodes_top[nx]])#[::-1]
# input_nodes = np.array(nodes_top[nx])

input_dofs = domain.get_dofnumber(input_nodes).ravel()

all_dofs = np.arange(0, 2*domain.nnodes)
main_dofs = np.concatenate([input_dofs, output_dofs])
free_dofs = np.setdiff1d(all_dofs, np.concatenate([fixed_dofs, main_dofs]))

x = pym.Signal('x', state=xmin*np.ones(domain.nel))

NMIMO = len(output_dofs)

domain2 = pym.VoxelDomain(NMIMO, NMIMO)
domain3 = pym.VoxelDomain(NMIMO, NMIMO)

with pym.Network() as fn:

  x_filtered = pym.DensityFilter(domain, radius=2)(x)
  vol = pym.EinSum('i->')(x_filtered)
  vol1 = pym.MathExpression(f'inp0/{nx*ny}')(vol)
  volcon = pym.MathExpression(f'inp0/{nx*ny} - {vf}')(vol)
  pym.PlotDomain(domain, saveto = timestamp + '/domain')(x_filtered)
  x_simp = pym.MathExpression(f'{xmin} + {1 - xmin}*inp0**3')(x_filtered)
  K = pym.AssembleStiffness(domain)(x_simp)
  K_condensed = pym.StaticCondensation(main=main_dofs, free=free_dofs)(K)
  C = pym.Inverse()(K_condensed)

  print('C', C.state)

  top = C[:NMIMO, :NMIMO]
  bot = C[NMIMO:, :NMIMO]

  print('top', top.state)
  print('bot', bot.state)

  top2 = pym.EinSum('ij,ij->ij')(top, top)
  bot2 = pym.EinSum('ij,ij->ij')(bot, bot)

  print('top2', top2.state)
  print('bot2', bot2.state)

  pym.PlotDomain(domain2)(top2)
  pym.PlotDomain(domain3)(bot2)

  top2sum = pym.EinSum('ij->')(top2)
  bot2sum = pym.EinSum('ij->')(bot2)

  topdiag = [top2[i, i] for i in range(NMIMO)]
  botdiag = [bot2[i, i] for i in range(NMIMO)]

  smallesttopdiag = pym.MathExpression('(1/inp0 + 1/inp1 + 1/inp2 + 1/inp3 + 1/inp4 + 1/inp5)**-.5')(topdiag[0], topdiag[1], topdiag[2], topdiag[3], topdiag[4], topdiag[5])

  topdiag2sum = pym.MathExpression('inp0 + inp1 + inp2 + inp3 + inp4 + inp5')(topdiag[0], topdiag[1], topdiag[2], topdiag[3], topdiag[4], topdiag[5])

  largesttopdiag = pym.MathExpression('inp0**.5')(topdiag2sum)

  smallestbotdiag = pym.MathExpression('(1/inp0 + 1/inp1 + 1/inp2 + 1/inp3 + 1/inp4 + 1/inp5)**-.5')(botdiag[0], botdiag[1], botdiag[2], botdiag[3], botdiag[4], botdiag[5])

  botdiag2sum = pym.MathExpression('inp0 + inp1 + inp2 + inp3 + inp4 + inp5')(botdiag[0], botdiag[1], botdiag[2], botdiag[3], botdiag[4], botdiag[5])

  largestbotdiag = pym.MathExpression('inp0**.5')(botdiag2sum)

  topoff2sum = pym.MathExpression('inp0 - inp1')(top2sum, topdiag2sum)
  botoff2sum = pym.MathExpression('inp0 - inp1')(bot2sum, botdiag2sum)

  largestoff = pym.MathExpression('(inp0 + inp1)**.5')(topoff2sum, botoff2sum)

  print(largesttopdiag.state, smallesttopdiag.state)
  print(largestbotdiag.state, smallestbotdiag.state)
  print(largestoff.state)

  obj = pym.MathExpression('inp0/inp1')(largestoff, smallestbotdiag)

  # objcon = [obj, volcon
  objcon =[obj]
  pym.PlotIter()(*objcon)
pym.minimize_mma(x, objcon, maxit=1000, tolx=0, tolf=0)

