
"""Utility functions for dealing with human3.6m data."""

from __future__ import division

import os
import numpy as np
import cameras
import h5py
import glob
import copy
import random


# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1,5,6,7,8]
TEST_SUBJECTS  = [9,11]

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*16
SH_NAMES[0]  = 'RFoot'
SH_NAMES[1]  = 'RKnee'
SH_NAMES[2]  = 'RHip'
SH_NAMES[3]  = 'LHip'
SH_NAMES[4]  = 'LKnee'
SH_NAMES[5]  = 'LFoot'
SH_NAMES[6]  = 'Hip'
SH_NAMES[7]  = 'Spine'
SH_NAMES[8]  = 'Thorax'
SH_NAMES[9]  = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'

def load_data( bpath, subjects, actions, dim=3 ):
  """
  Loads 2d ground truth from disk, and puts it in an easy-to-acess dictionary

  Args
    bpath: String. Path where to load the data from
    subjects: List of integers. Subjects whose data will be loaded
    actions: List of strings. The actions to load
    dim: Integer={2,3}. Load 2 or 3-dimensional data
  Returns:
    data: Dictionary with keys k=(subject, action, seqname)
      values v=(nx(32*2) matrix of 2d ground truth)
      There will be 2 entries per subject/action if loading 3d data
      There will be 8 entries per subject/action if loading 2d data
  """

  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  data = {}

  for subj in subjects:
    for action in actions:

      print('Reading subject {0}, action {1}'.format(subj, action))

      dpath = os.path.join( bpath, 'S{0}'.format(subj), 'MyPoses/{0}D_positions'.format(dim), '{0}*.h5'.format(action) )
      print( dpath )

      fnames = glob.glob( dpath )

      loaded_seqs = 0
      for fname in fnames:
        seqname = os.path.basename( fname )

        # This rule makes sure SittingDown is not loaded when Sitting is requested
        if action == "Sitting" and seqname.startswith( "SittingDown" ):
          continue

        # This rule makes sure that WalkDog and WalkTogeter are not loaded when
        # Walking is requested.
        if seqname.startswith( action ):
          print( fname )
          loaded_seqs = loaded_seqs + 1

          with h5py.File( fname, 'r' ) as h5f:
            poses = h5f['{0}D_positions'.format(dim)][:]

          poses = poses.T
          data[ (subj, action, seqname) ] = poses

      if dim == 2:
        assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead".format( loaded_seqs )
      else:
        assert loaded_seqs == 2, "Expecting 2 sequences, found {0} instead".format( loaded_seqs )

  return data


def load_stacked_hourglass(data_dir, subjects, actions):
  """
  Load 2d detections from disk, and put it in an easy-to-acess dictionary.

  Args
    data_dir: string. Directory where to load the data from,
    subjects: list of integers. Subjects whose data will be loaded.
    actions: list of strings. The actions to load.
  Returns
    data: dictionary with keys k=(subject, action, seqname)
          values v=(nx(32*2) matrix of 2d stacked hourglass detections)
          There will be 2 entries per subject/action if loading 3d data
          There will be 8 entries per subject/action if loading 2d data
  """
  # Permutation that goes from SH detections to H36M ordering.
  SH_TO_GT_PERM = np.array([SH_NAMES.index( h ) for h in H36M_NAMES if h != '' and h in SH_NAMES])
  assert np.all( SH_TO_GT_PERM == np.array([6,2,1,0,3,4,5,7,8,9,13,14,15,12,11,10]) )

  data = {}

  for subj in subjects:
    for action in actions:

      print('Reading subject {0}, action {1}'.format(subj, action))

      dpath = os.path.join( data_dir, 'S{0}'.format(subj), 'StackedHourglassFineTuned240/{0}*.h5'.format(action) )
      print( dpath )

      fnames = glob.glob( dpath )

      loaded_seqs = 0
      for fname in fnames:
        seqname = os.path.basename( fname )
        seqname = seqname.replace('_',' ')

        # This rule makes sure SittingDown is not loaded when Sitting is requested
        if action == "Sitting" and seqname.startswith( "SittingDown" ):
          continue

        # This rule makes sure that WalkDog and WalkTogeter are not loaded when
        # Walking is requested.
        if seqname.startswith( action ):
          print( fname )
          loaded_seqs = loaded_seqs + 1

          # Load the poses from the .h5 file
          with h5py.File( fname, 'r' ) as h5f:
            poses = h5f['poses'][:]

            # Permute the loaded data to make it compatible with H36M
            poses = poses[:,SH_TO_GT_PERM,:]

            # Reshape into n x (32*2) matrix
            poses = np.reshape(poses,[poses.shape[0], -1])
            poses_final = np.zeros([poses.shape[0], len(H36M_NAMES)*2])

            dim_to_use_x    = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0] * 2
            dim_to_use_y    = dim_to_use_x+1

            dim_to_use = np.zeros(len(SH_NAMES)*2,dtype=np.int32)
            dim_to_use[0::2] = dim_to_use_x
            dim_to_use[1::2] = dim_to_use_y
            poses_final[:,dim_to_use] = poses
            seqname = seqname+'-sh'
            data[ (subj, action, seqname) ] = poses_final

      # Make sure we loaded 8 sequences
      if (subj == 11 and action == 'Directions'): # <-- this video is damaged
        assert loaded_seqs == 7, "Expecting 7 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )
      else:
        assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )

  return data


def normalization_stats(complete_data, dim, predict_14=False ):
  """
  Computes normalization statistics: mean and stdev, dimensions used and ignored

  Args
    complete_data: nxd np array with poses
    dim. integer={2,3} dimensionality of the data
    predict_14. boolean. Whether to use only 14 joints
  Returns
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions not used in the model
    dimensions_to_use: list of dimensions used in the model
  """
  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  data_mean = np.mean(complete_data, axis=0)
  data_std  =  np.std(complete_data, axis=0)

  # Encodes which 17 (or 14) 2d-3d pairs we are predicting
  dimensions_to_ignore = []
  if dim == 2:
    dimensions_to_use    = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
    dimensions_to_use    = np.sort( np.hstack( (dimensions_to_use*2, dimensions_to_use*2+1)))
    dimensions_to_ignore = np.delete( np.arange(len(H36M_NAMES)*2), dimensions_to_use )
  else: # dim == 3
    dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
    dimensions_to_use = np.delete( dimensions_to_use, [0,7,9] if predict_14 else 0 )

    dimensions_to_use = np.sort( np.hstack( (dimensions_to_use*3,
                                             dimensions_to_use*3+1,
                                             dimensions_to_use*3+2)))
    dimensions_to_ignore = np.delete( np.arange(len(H36M_NAMES)*3), dimensions_to_use )

  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def transform_world_to_camera(poses_set, cams, ncams=4 ):
    """
    Project 3d poses from world coordinate to camera coordinate system
    Args
      poses_set: dictionary with 3d poses
      cams: dictionary with cameras
      ncams: number of cameras per subject
    Return:
      t3d_camera: dictionary with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for t3dk in sorted( poses_set.keys() ):

      subj, action, seqname = t3dk
      t3d_world = poses_set[ t3dk ]

      for c in range( ncams ):
        R, T, f, c, k, p, name = cams[ (subj, c+1) ]
        camera_coord = cameras.world_to_camera_frame( np.reshape(t3d_world, [-1, 3]), R, T)
        camera_coord = np.reshape( camera_coord, [-1, len(H36M_NAMES)*3] )

        sname = seqname[:-3]+"."+name+".h5" # e.g.: Waiting 1.58860488.h5
        t3d_camera[ (subj, action, sname) ] = camera_coord

    return t3d_camera


def normalize_data(data, data_mean, data_std, dim_to_use ):
  """
  Normalizes a dictionary of poses

  Args
    data: dictionary where values are
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dim_to_use: list of dimensions to keep in the data
  Returns
    data_out: dictionary with same keys as data, but values have been normalized
  """
  data_out = {}

  for key in data.keys():
    data[ key ] = data[ key ][ :, dim_to_use ]
    mu = data_mean[dim_to_use]
    stddev = data_std[dim_to_use]
    data_out[ key ] = np.divide( (data[key] - mu), stddev )

  return data_out


def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_ignore):
  """
  Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing

  Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions that were removed from the original data
  Returns
    orig_data: the input normalized_data, but unnormalized
  """
  T = normalized_data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality

  orig_data = np.zeros((T, D), dtype=np.float32)
  dimensions_to_use = np.array([dim for dim in range(D)
                                if dim not in dimensions_to_ignore])

  orig_data[:, dimensions_to_use] = normalized_data

  # Multiply times stdev and add the mean
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  orig_data = np.multiply(orig_data, stdMat) + meanMat
  return orig_data


def define_actions( action ):
  """
  Given an action string, returns a list of corresponding actions.

  Args
    action: String. either "all" or one of the h36m actions
  Returns
    actions: List of strings. Actions to use.
  Raises
    ValueError: if the action is not a valid action in Human 3.6M
  """
  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all":
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]


def project_to_cameras( poses_set, cams, ncams=4 ):
  """
  Project 3d poses using camera parameters

  Args
    poses_set: dictionary with 3d poses
    cams: dictionary with camera parameters
    ncams: number of cameras per subject
  Returns
    t2d: dictionary with 2d poses
  """
  t2d = {}

  for t3dk in sorted( poses_set.keys() ):
    subj, a, seqname = t3dk
    t3d = poses_set[ t3dk ]

    for cam in range( ncams ):
      R, T, f, c, k, p, name = cams[ (subj, cam+1) ]
      pts2d, _, _, _, _ = cameras.project_point_radial( np.reshape(t3d, [-1, 3]), R, T, f, c, k, p )

      pts2d = np.reshape( pts2d, [-1, len(H36M_NAMES)*2] )
      sname = seqname[:-3]+"."+name+".h5" # e.g.: Waiting 1.58860488.h5
      t2d[ (subj, a, sname) ] = pts2d

  return t2d


def read_2d_predictions( actions, data_dir ):
  """
  Loads 2d data from precomputed Stacked Hourglass detections

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
  Returns
    train_set: dictionary with loaded 2d stacked hourglass detections for training
    test_set: dictionary with loaded 2d stacked hourglass detections for testing
    data_mean: vector with the mean of the 2d training data
    data_std: vector with the standard deviation of the 2d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  """

  train_set = load_stacked_hourglass( data_dir, TRAIN_SUBJECTS, actions)
  test_set  = load_stacked_hourglass( data_dir, TEST_SUBJECTS,  actions)

  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std,  dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )

  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def create_2d_data( actions, data_dir, rcams ):
  """
  Creates 2d poses by projecting 3d poses with the corresponding camera
  parameters. Also normalizes the 2d poses

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
    rcams: dictionary with camera parameters
  Returns
    train_set: dictionary with projected 2d poses for training
    test_set: dictionary with projected 2d poses for testing
    data_mean: vector with the mean of the 2d training data
    data_std: vector with the standard deviation of the 2d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  """

  # Load 3d data
  train_set = load_data( data_dir, TRAIN_SUBJECTS, actions, dim=3 )
  test_set  = load_data( data_dir, TEST_SUBJECTS,  actions, dim=3 )

  train_set = project_to_cameras( train_set, rcams )
  test_set  = project_to_cameras( test_set, rcams )


  # Compute normalization statistics.
  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def read_3d_data( actions, data_dir, camera_frame, rcams, predict_14=False ):
  """
  Loads 3d poses, zero-centres and normalizes them

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
    camera_frame: boolean. Whether to convert the data to camera coordinates
    rcams: dictionary with camera parameters
    predict_14: boolean. Whether to predict only 14 joints
  Returns
    train_set: dictionary with loaded 3d poses for training
    test_set: dictionary with loaded 3d poses for testing
    data_mean: vector with the mean of the 3d training data
    data_std: vector with the standard deviation of the 3d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
    train_root_positions: dictionary with the 3d positions of the root in train
    test_root_positions: dictionary with the 3d positions of the root in test
  """
  # Load 3d data
  train_set = load_data( data_dir, TRAIN_SUBJECTS, actions, dim=3 )
  test_set  = load_data( data_dir, TEST_SUBJECTS,  actions, dim=3 )


  if camera_frame:
    train_set = transform_world_to_camera( train_set, rcams )
    test_set  = transform_world_to_camera( test_set, rcams )

  # Apply 3d post-processing (centering around root)
  train_set, train_root_positions = postprocess_3d( train_set )
  test_set,  test_root_positions  = postprocess_3d( test_set )

  # Compute normalization statistics
  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=3, predict_14=predict_14 )

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions


def postprocess_3d( poses_set ):
  """
  Center 3d points around root

  Args
    poses_set: dictionary with 3d data
  Returns
    poses_set: dictionary with 3d data centred around root (center hip) joint
    root_positions: dictionary with the original 3d position of each pose
  """
  root_positions = {}
  for k in poses_set.keys():
    # Keep track of the global position
    root_positions[k] = copy.deepcopy(poses_set[k][:,:3])

    # Remove the root from the 3d position
    poses = poses_set[k]
    poses = poses - np.tile( poses[:,:3], [1, len(H36M_NAMES)] )
    poses_set[k] = poses

  return poses_set, root_positions



def create_2d_mpii_test(dataset, Debug=False):

  '''
   Create 2d pose data as the input of the stage two.
   For mpii dataset, we use the output of the hourglass network
   For mpi dataset, we use the 2d joints provided by the dataset
   Args:
     dataset: spicify which dataset to use, either 'mpi' or 'mpii'

  '''

  mpii_to_human36 = np.array([6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 13, 14, 15, 12, 11, 10])

  if dataset == 'mpi':
    input_file = '../data/mpi/annotVal_outdoor.h5'  # mpi dataset has three scenario, green background, normal indoor and outdoor
    annot_train = getData(input_file)
    joints = annot_train['annot_2d']

  else:
    input_file = '../data/mpii/mpii_preds.h5'
    annot_train = getData(input_file)
    joints = annot_train['part']

  joints = joints[:, mpii_to_human36, :]      # only use the correspoinding joints

  dim_to_use_x = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0] * 2
  dim_to_use_y = dim_to_use_x + 1

  dim_to_use = np.zeros(len(SH_NAMES) * 2, dtype=np.int32)
  dim_to_use[0::2] = dim_to_use_x
  dim_to_use[1::2] = dim_to_use_y

  dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 2), dim_to_use)
  poses = np.reshape(joints, [joints.shape[0], -1])

  poses_final = np.zeros([poses.shape[0], len(H36M_NAMES) * 2])
  poses_final[:, dim_to_use] = poses

  print('{} left from {} after filter'.format(poses_final.shape[0], poses.shape[0]))
  complete_train = copy.deepcopy(poses_final)
  test_set_2d, data_mean, data_std = normalize_data_mpii(complete_train, dim_to_use)

  # if Debug:
  #
  #   data = unNormalizeData(train_set, data_mean, data_std, dimensions_to_ignore)
  #   for i in range(data.shape[0]):
  #     pose = data[i, dim_to_use].reshape(16,2)
  #     human36_to_mpii = np.argsort(mpii_to_human36)
  #     pose_mpii = pose[human36_to_mpii]
  #     name = names[i][:13]
  #     imgpath = '/home/lichen/pose_estimation/images_2d/'
  #     img = cv2.imread(os.path.join(imgpath, name))
  #     c = (255, 0, 0)
  #
  #     for j in range(pose_mpii.shape[0]):
  #       cv2.circle(img, (int(pose_mpii[j, 0]), int(pose_mpii[j, 1])), 3, c, -1)
  #     cv2.imshow('img', img)
  #     cv2.waitKey()

  return test_set_2d, data_mean, data_std, dim_to_use

def get_all_batches_mpii(data, batch_size):
  '''

  data: all data to use
  batch_size: batch size for model
  return: data in batch
  '''

  data = np.array(data)
  n = data.shape[0]
  n_extra = np.int32(n % batch_size)
  n_batches = np.int32(n // batch_size )
  if n_extra>0:
    encoder_inputs = np.split(data[:-n_extra, :],n_batches )
  else:
    encoder_inputs = np.split(data, n_batches)
  return encoder_inputs


def normalize_data_mpii(data, dim_to_use):
  """
  Normalize the 2d pose data
  Args
    data:
    dim_to_use: list of dimensions to keep in the 2d pose data
  Returns
    data_out: normalized data, mean and standard deviation
  """


  data_mean = np.mean(data, axis=0)
  data_std = np.std(data,axis=0)
  data_out = np.divide( (data[:, dim_to_use] - data_mean[dim_to_use]), data_std[dim_to_use] )

  return data_out, data_mean, data_std

def unnormalize_data_mpii(normalized_data, data_mean, data_std, dimensions_to_use):

  '''
    Unnormalize the 2d pose data
  '''

  T = normalized_data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality

  orig_data = np.zeros((T, D), dtype=np.float32)
  orig_data[:, dimensions_to_use] = normalized_data

  # Multiply times stdev and add the mean
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  orig_data = np.multiply(orig_data, stdMat) + meanMat

  return orig_data

def h36_to_mpii(pose):
  h36_to_mpii_permu = np.array([3, 2, 1, 4, 5, 6, 0, 7, 8, 9, 15, 14, 13, 10, 11, 12])   # joint indexes for mpii dataset and h36 are different
  pose = np.reshape(pose, [pose.shape[0], len(SH_NAMES), -1])
  pose = pose[:, h36_to_mpii_permu]

  return pose

def create_3d_mpi_test():

  '''
  Create 3d pose data for mpi data set
  '''

  mpii_to_human36 = np.array([6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 13, 14, 15, 12, 11, 10])   # to make the joint index in consistence
  input_file = '../data/mpi/annotVal_outdoor.h5'
  data = h5py.File(input_file, 'r')
  joints_3d = data['annot_3d'].value
  img_name = data['annot_image'].value

  poses = joints_3d[:, mpii_to_human36, :]     #  mpi does not have the annotation for joint 'neck', approximate by the average value of throat and head
  pose_neck = (poses[:, 8, :] + poses[:, 10, :])/2
  poses_17 = np.insert(poses, 9 , pose_neck, axis= 1)
  poses_mpi_to_h36 = copy.deepcopy( poses_17)

  poses_17 = np.reshape(poses_17, [poses_17.shape[0], -1])
  poses_final = np.zeros([poses.shape[0], len(H36M_NAMES) * 3])


  dim_to_use_x = np.where(np.array([x != '' for x in H36M_NAMES]))[0] * 3
  dim_to_use_y = dim_to_use_x + 1
  dim_to_use_z = dim_to_use_x + 2

  dim_to_use = np.zeros(17 * 3, dtype=np.int32)
  dim_to_use[0::3] = dim_to_use_x
  dim_to_use[1::3] = dim_to_use_y
  dim_to_use[2::3] = dim_to_use_z
  poses_final[:, dim_to_use] = poses_17

  test_set, test_root_positions = postprocess_3d_mpi(poses_final)
  complete_test = copy.deepcopy(np.vstack(test_set))
  data_mean, data_std, dim_to_ignore, dim_to_use_ = normalization_stats(complete_test, dim=3)

  # Divide every dimension independently
  test_set = normalize_data_mpi(test_set, data_mean, data_std, dim_to_use_)
  return  test_set, data_mean, data_std, dim_to_ignore, dim_to_use_, test_root_positions, poses_mpi_to_h36, img_name

def postprocess_3d_mpi(pose_3d):
  '''

   process the 3d pose data with respect to the root joint
   We regress the relative rather than the absolute coordinates,
  '''

  root_position = copy.deepcopy(pose_3d[:, :3])

  pose_3d_root = pose_3d - np.tile(root_position,[1, len(H36M_NAMES)])

  return pose_3d_root, root_position

def normalize_data_mpi(data, mean, std, dim_to_use):

  '''
    Normalize the 3d pose data in mpi dataset
  '''

  data = data[:, dim_to_use]
  mean = mean[dim_to_use]
  std = std[dim_to_use]
  data_out = np.divide((data-mean), std+0.0000001)

  return data_out

def getData(tmpFile):
  '''
    Read data from .h5 file
  '''
  data = h5py.File(tmpFile, 'r')
  d = {}
  for k, v in data.items():
    d[k] = np.asarray(data[k])
  data.close()
  return d


def generage_missing_data(enc_in,mis_number):
  '''

  enc_in: input 2d pose data
  mis_number: the number of missing joints
  return: 2d pose with missing joints randomly selected from the limbs
  '''

  joints_missing = [2, 3, 5, 6, 11, 12, 14, 15]    # only delete joints from limbs
  for i in range(enc_in.shape[0]):
    if mis_number == 1:
      missing_index = random.randint(0, 7)
      missing_dim = np.array([joints_missing[missing_index]*2, joints_missing[missing_index]*2+1])
    else:
      missing_index = random.sample(range(8), 2)
      missing_dim = np.array([joints_missing[missing_index[0]] * 2, joints_missing[missing_index[0]] * 2 + 1,
                              joints_missing[missing_index[1]] * 2, joints_missing[missing_index[1]] * 2 + 1])

    enc_in[i, missing_dim] = 0.0       # get missing joints by setting the corresponding value to 0

  return enc_in










