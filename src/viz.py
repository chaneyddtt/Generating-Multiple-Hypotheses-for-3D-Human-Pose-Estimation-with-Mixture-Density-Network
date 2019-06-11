
"""Functions to visualize human poses"""

import matplotlib.pyplot as plt
import data_utils
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D

def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False): # blue, orange
  """
  Visualize the ground truth 3d skeleton

  Args
    channels: 96x1 vector. The pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (len(data_utils.H36M_NAMES), -1) )

  I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
  J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
  LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
    # ax.plot(x, y, z, lw=3, marker = 'o', markersize = 5, c=lcolor if LR[i] else rcolor, markeredgecolor = lcolor)
    ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

  RADIUS = 750 # space around the subject
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

  # ax.set_xlim3d([np.min(vals[:, 0]), np.max(vals[:, 0])])
  # ax.set_zlim3d([np.min(vals[:, 2]), np.max(vals[:, 2])])
  # ax.set_ylim3d([np.min(vals[:, 1]), np.max(vals[:, 1])])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

  # Get rid of the ticks and tick labels

  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])

  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])
  ax.set_zticklabels([])
  ax.set_aspect('equal')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  # ax.w_zaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # # Keep z pane
  #
  # # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)


  ax.view_init(azim=129, elev=10)



def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
  """
  Visualize a 2d skeleton with 32 joints

  Args
    channels: 64x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  assert channels.size == len(data_utils.H36M_NAMES)*2, "channels should have 64 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (len(data_utils.H36M_NAMES), -1) )

  I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
  J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
  LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
    ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

  # Get rid of the ticks
  ax.set_xticks([])
  ax.set_yticks([])

  # Get rid of tick labels
  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])

  RADIUS = 300 # space around the subject
  xroot, yroot = vals[0,0], vals[0,1]
  ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])
  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")

  ax.set_aspect('equal')




def show2Dpose_mdm(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
  """
  Visualize 2d reprojections of all 3d pose hypotheses in one fig in order to show the similarity between them

  Args
    channels: 64 * 5, 2d reprojections of all 3d pose hypotheses
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body. Note that we do not really use lcolor and rcolor in this function. 
	        In stead, we define a color for each hypotheses to show the overlap between them.
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """




  I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
  J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
  LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
  colors = ['#FF8000', '#4169E1', '#308014', '#000080', '#FF83FA'] # color used for 2d reprejection from each 3d pose hypotheses 
  for m in range(channels.shape[-1]):
    vals = np.reshape(channels[:,m], [len(data_utils.H36M_NAMES), -1])
    for i in np.arange( len(I) ):
      x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
      ax.plot(x, y, lw=2, c=colors[m])

  # Get rid of the ticks
  ax.set_xticks([])
  ax.set_yticks([])

  # Get rid of tick labels
  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])

  RADIUS = 300 # space around the subject
  xroot, yroot = vals[0,0], vals[0,1]
  ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])
  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")

  ax.set_aspect('equal')
