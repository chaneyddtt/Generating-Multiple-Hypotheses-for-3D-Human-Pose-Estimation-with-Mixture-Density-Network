
"""Predicting 3d poses from 2d joints"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py
import copy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import procrustes

import viz
import cameras
import data_utils
import mix_den_model
import logging, logging.config


tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability. 1 means no dropout")
tf.app.flags.DEFINE_integer("batch_size", 64,"batch size to use during training")
tf.app.flags.DEFINE_integer("epochs", 200, "How many epochs we should train for")
tf.app.flags.DEFINE_boolean("camera_frame", True, "Convert 3d poses to camera coordinates")
tf.app.flags.DEFINE_boolean("max_norm", True, "Apply maxnorm constraint to the weights")
tf.app.flags.DEFINE_boolean("batch_norm", True, "Use batch_normalization")

# Data loading
tf.app.flags.DEFINE_boolean("predict_14", False, "predict 14 joints")
tf.app.flags.DEFINE_boolean("use_sh", True, "Use 2d pose predictions from StackedHourglass")
tf.app.flags.DEFINE_string("action","All", "The action to train on. 'All' means all the actions")

# Architecture
tf.app.flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("residual", True, "Whether to add a residual connection every 2 layers")

# Evaluation
tf.app.flags.DEFINE_boolean("procrustes", False, "Apply procrustes analysis at test time")
tf.app.flags.DEFINE_boolean("evaluateActionWise",True, "The dataset to use either h36m or heva")

# Directories
tf.app.flags.DEFINE_string("cameras_path","../data/h36m/cameras.h5","Directory to load camera parameters")
tf.app.flags.DEFINE_string("data_dir",   "../data/h36m/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "../experiments/test_git/", "Training directory.")
tf.app.flags.DEFINE_string("load_dir", "../Models/mdm_5_prior/", "Specify the directory to load trained model")

# Train or load
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("test", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")
tf.app.flags.DEFINE_integer("miss_num", 1, "Specify how many missing joints.")

### 4679232 for mdm_5
### 4338038 for mdm prior

# Misc
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

def make_dir_if_not_exist(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

train_dir = FLAGS.train_dir
load_dir = FLAGS.load_dir
summaries_dir = os.path.join( train_dir, "summary" )
logdir = os.path.join(train_dir,"log")
os.system('mkdir -p {}'.format(summaries_dir))
make_dir_if_not_exist(logdir)

logging.config.fileConfig('./logging.conf')
logger = logging.getLogger()
fileHandler = logging.FileHandler("{0}/log.txt".format(logdir))
logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)
logger.info("Logs will be written to %s" % logdir)







def create_model( session, actions, batch_size ):
  """
  Create model and initialize it or load its parameters in a session

  Args
    session: tensorflow session
    actions: list of string. Actions to train/test on
    batch_size: integer. Number of examples in each batch
  Returns
    model: The created (or loaded) model
  Raises
    ValueError if asked to load a model, but the checkpoint specified by
    FLAGS.load cannot be found.
  """

  model = mix_den_model.LinearModel(
      FLAGS.linear_size,
      FLAGS.num_layers,
      FLAGS.residual,
      FLAGS.batch_norm,
      FLAGS.max_norm,
      batch_size,
      FLAGS.learning_rate,
      summaries_dir,
      FLAGS.predict_14,
      dtype=tf.float16 if FLAGS.use_fp16 else tf.float32)

  if FLAGS.load <= 0:
    # Create a new model from scratch
    print("Creating model with fresh parameters.")
    session.run( tf.global_variables_initializer() )
    return model

  # Load a previously saved model
  ckpt = tf.train.get_checkpoint_state( load_dir, latest_filename="checkpoint")
  print( "train_dir", load_dir )

  if ckpt and ckpt.model_checkpoint_path:
    # Check if the specific checkpoint exists
    if FLAGS.load > 0:
      if os.path.isfile(os.path.join(load_dir,"checkpoint-{0}.index".format(FLAGS.load))):
        ckpt_name = os.path.join( os.path.join(load_dir,"checkpoint-{0}".format(FLAGS.load)) )
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    model.saver.restore( session, ckpt_name )
    return model
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )


def train():
  """Train a linear model for 3d pose estimation"""

  actions = data_utils.define_actions( FLAGS.action )

  # Load camera parameters
  SUBJECT_IDS = [1,5,6,7,8,9,11]
  rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)

  # Load 3d data and load (or create) 2d projections
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
    actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14 )

  # Read stacked hourglass 2D predictions if use_sh, otherwise use groundtruth 2D projections
  if FLAGS.use_sh:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, FLAGS.data_dir)
  else:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions, FLAGS.data_dir, rcams )


  # Avoid using the GPU if requested
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto(
    device_count=device_count,
    allow_soft_placement=True )) as sess:

    # === Create the model ===
    print("Creating %d bi-layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    model = create_model( sess, actions, FLAGS.batch_size )
    model.train_writer.add_graph( sess.graph )


    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1


    current_epoch = 0
    log_every_n_batches = 100


    for epoch in xrange( FLAGS.epochs ):
      current_epoch = current_epoch + 1

      # === Load training batches for one epoch ===
      encoder_inputs, decoder_outputs, _ = model.get_all_batches( train_set_2d, train_set_3d, FLAGS.camera_frame, training=True )
      nbatches = len( encoder_inputs )
      start_time, loss = time.time(), 0.

      # === Loop through all the training batches ===
      for i in range( nbatches):

        if (i+1) % log_every_n_batches == 0:
          # Print progress every log_every_n_batches batches
          print("Working on epoch {0}, batch {1} / {2}... ".format( current_epoch, i+1, nbatches), end="" )

        enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
        # enc_in = data_utils.generage_missing_data(enc_in, FLAGS.miss_num)
        step_loss, loss_summary, lr_summary, comp =  model.step( sess, enc_in, dec_out, FLAGS.dropout, isTraining=True )


        if (i+1) % log_every_n_batches == 0:

          # Log and print progress every log_every_n_batches batches

          model.train_writer.add_summary( loss_summary, current_step )
          model.train_writer.add_summary( lr_summary, current_step )
          step_time = (time.time() - start_time)
          start_time = time.time()
          print("done in {0:.2f} ms".format( 1000*step_time / log_every_n_batches ) )

        loss += step_loss
        current_step += 1
        # === end looping through training batches ===

      loss = loss / nbatches

      logger.info("=============================\n"
            "Epoch:               %d\n"
            "Global step:         %d\n"
            "Learning rate:       %.2e\n"
            "Train loss avg:      %.4f\n"
            "=============================" % (epoch, model.global_step.eval(),
            model.learning_rate.eval(), loss) )
      # === End training for an epoch ===

      # === Testing after this epoch ===

      if FLAGS.evaluateActionWise:

        logger.info("{0:=^12} {1:=^6}".format("Action", "mm")) # line of 30 equal signs

        cum_err = 0           # select the mixture model which has mininum error
        for action in actions:


          # Get 2d and 3d testing data for this action
          action_test_set_2d = get_action_subset( test_set_2d, action )
          action_test_set_3d = get_action_subset( test_set_3d, action )
          encoder_inputs, decoder_outputs, repro_info = model.get_all_batches( action_test_set_2d, action_test_set_3d, FLAGS.camera_frame, training=False)

          act_err, step_time, loss = evaluate_batches( sess, model,
                                                      data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
                                                      data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
                                                      current_step, encoder_inputs, decoder_outputs)

          cum_err = cum_err + act_err
          logger.info('{0:<12} {1:>6.2f}'.format(action, act_err))

        summaries = sess.run( model.err_mm_summary, {model.err_mm: float(cum_err/float(len(actions)))} )
        model.test_writer.add_summary( summaries, current_step )

        logger.info('{0:<12} {1:>6.2f}'.format("Average", cum_err/float(len(actions))))

        logger.info('{0:=^19}'.format(''))

      # Save the model
      print( "Saving the model... ", end="" )
      start_time = time.time()
      if cum_err/float(len(actions))<60.66:
        model.saver.save(sess, os.path.join(train_dir, 'checkpoint'), global_step=current_step)
      print( "done in {0:.2f} ms".format(1000*(time.time() - start_time)) )

      # Reset global time and loss
      step_time, loss = 0, 0

      sys.stdout.flush()


def get_action_subset( poses_set, action ):
  """
  Given a preloaded dictionary of poses, load the subset of a particular action

  Args
    poses_set: dictionary with keys k=(subject, action, seqname),
      values v=(nxd matrix of poses)
    action: string. The action that we want to filter out
  Returns
    poses_subset: dictionary with same structure as poses_set, but only with the
      specified action.
  """
  return {k:v for k, v in poses_set.items() if k[1] == action}


def evaluate_batches( sess, model,
  data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
  data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
  current_step, encoder_inputs, decoder_outputs, current_epoch=0 ):
  """
  Generic method that evaluates performance of a list of batches.
  May be used to evaluate all actions or a single action.

  Args
    sess
    model
    data_mean_3d
    data_std_3d
    dim_to_use_3d
    dim_to_ignore_3d
    data_mean_2d
    data_std_2d
    dim_to_use_2d
    dim_to_ignore_2d
    current_step
    encoder_inputs
    decoder_outputs
    current_epoch
  Returns

    total_err
    joint_err
    step_time
    loss
  """

  n_joints = 17 if not(FLAGS.predict_14) else 14
  nbatches = len( encoder_inputs )


  # Loop through test examples
  all_dists, start_time, loss = [], time.time(), 0.
  log_every_n_batches = 100
  all_poses_3d = []
  all_enc_in =[]

  for i in range(nbatches):

    if current_epoch > 0 and (i+1) % log_every_n_batches == 0:
      print("Working on test epoch {0}, batch {1} / {2}".format( current_epoch, i+1, nbatches) )

    enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
    # enc_in = data_utils.generage_missing_data(enc_in, FLAGS.miss_num)
    dp = 1.0 # dropout keep probability is always 1 at test time
    step_loss, loss_summary, out_all_components_ori = model.step( sess, enc_in, dec_out, dp, isTraining=False )
    loss += step_loss

    out_all_components = np.reshape(out_all_components_ori,[-1, model.HUMAN_3D_SIZE+2, model.num_models])
    out_mean = out_all_components[:, : model.HUMAN_3D_SIZE, :]


    # denormalize
    enc_in  = data_utils.unNormalizeData( enc_in,  data_mean_2d, data_std_2d, dim_to_ignore_2d )
    enc_in_ = copy.deepcopy(enc_in)
    all_enc_in.append(enc_in_)
    dec_out = data_utils.unNormalizeData( dec_out, data_mean_3d, data_std_3d, dim_to_ignore_3d )
    pose_3d = np.zeros((enc_in.shape[0],96, out_mean.shape[-1]))

    for j in range(out_mean.shape[-1]):
        pose_3d[:, :, j] = data_utils.unNormalizeData( out_mean[:, :, j], data_mean_3d, data_std_3d, dim_to_ignore_3d )

    pose_3d_ = copy.deepcopy(pose_3d)
    all_poses_3d.append(pose_3d_)

    # Keep only the relevant dimensions
    dtu3d = np.hstack( (np.arange(3), dim_to_use_3d) ) if not(FLAGS.predict_14) else  dim_to_use_3d

    dec_out = dec_out[:, dtu3d]
    pose_3d = pose_3d[:, dtu3d,:]

    assert dec_out.shape[0] == FLAGS.batch_size
    assert pose_3d.shape[0] == FLAGS.batch_size

    if FLAGS.procrustes:
      # Apply per-frame procrustes alignment if asked to do so
      for j in range(FLAGS.batch_size):
          for k in range(model.num_models):
            gt  = np.reshape(dec_out[j,:],[-1,3])
            out = np.reshape(pose_3d[j,:, k],[-1,3])
            _, Z, T, b, c = procrustes.compute_similarity_transform(gt,out,compute_optimal_scale=True)
            out = (b*out.dot(T))+c

            pose_3d[j, :, k] = np.reshape(out,[-1,17*3] ) if not(FLAGS.predict_14) else np.reshape(pose_3d[j,:, k],[-1,14*3] )

    # Compute Euclidean distance error per joint
    sqerr = (pose_3d - np.expand_dims(dec_out,axis=2))**2 # Squared error between prediction and expected output
    dists = np.zeros((sqerr.shape[0], n_joints, sqerr.shape[2])) # Array with L2 error per joint in mm

    for m in range(dists.shape[-1]):
      dist_idx = 0
      for k in np.arange(0, n_joints*3, 3):
        # Sum across X,Y, and Z dimenstions to obtain L2 distance
        dists[:,dist_idx, m] = np.sqrt( np.sum( sqerr[:, k:k+3,m], axis=1 ))

        dist_idx = dist_idx + 1

    all_dists.append(dists)
    assert sqerr.shape[0] == FLAGS.batch_size

  step_time = (time.time() - start_time) / nbatches
  loss      = loss / nbatches

  all_dists = np.vstack( all_dists )
  aver_minerr = np.mean(np.min(np.sum( all_dists, axis=1),axis=1))/n_joints

  return aver_minerr, step_time, loss


def test():

  actions = data_utils.define_actions( FLAGS.action )

  # Load camera parameters
  SUBJECT_IDS = [1,5,6,7,8,9,11]
  rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)

  # Load 3d data and load (or create) 2d projections
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
    actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14 )

  # Read stacked hourglass 2D predictions if use_sh, otherwise use groundtruth 2D projections
  if FLAGS.use_sh:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, FLAGS.data_dir)
  else:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions, FLAGS.data_dir, rcams )


  # Avoid using the GPU if requested
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto(
    device_count=device_count,
    allow_soft_placement=True )) as sess:

    # === Create the model ===
    print("Creating %d bi-layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    model = create_model( sess, actions, FLAGS.batch_size )
    model.train_writer.add_graph( sess.graph )

    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1

    if FLAGS.evaluateActionWise:

      logger.info("{0:=^12} {1:=^6}".format("Action", "mm")) # line of 30 equal signs

      cum_err = 0           # select the mixture model which has mininum error
      for action in actions:


        # Get 2d and 3d testing data for this action
        action_test_set_2d = get_action_subset( test_set_2d, action )
        action_test_set_3d = get_action_subset( test_set_3d, action )
        encoder_inputs, decoder_outputs, repro_info = model.get_all_batches( action_test_set_2d, action_test_set_3d, FLAGS.camera_frame, training=False)

        act_err, step_time, loss = evaluate_batches( sess, model,
          data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
          data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
          current_step, encoder_inputs, decoder_outputs)

        cum_err = cum_err + act_err
        logger.info('{0:<12} {1:>6.2f}'.format(action, act_err))

      summaries = sess.run( model.err_mm_summary, {model.err_mm: float(cum_err/float(len(actions)))} )
      model.test_writer.add_summary( summaries, current_step )

      logger.info('{0:<12} {1:>6.2f}'.format("Average", cum_err/float(len(actions))))

      logger.info('{0:=^19}'.format(''))


def sample():

  """Get samples from a model and visualize them"""
  path = '{}/samples_sh'.format(FLAGS.train_dir)
  if not os.path.exists(path):
    os.makedirs(path)
  actions = data_utils.define_actions( FLAGS.action )

  # Load camera parameters
  SUBJECT_IDS = [1,5,6,7,8,9,11]
  rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)
  n_joints = 17 if not (FLAGS.predict_14) else 14

  # Load 3d data and load (or create) 2d projections
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
    actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14 )

  if FLAGS.use_sh:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, FLAGS.data_dir)
  else:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d, _ = data_utils.create_2d_data( actions, FLAGS.data_dir, rcams )

  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto( device_count = device_count )) as sess:
    # === Create the model ===

    batch_size = 128
    model = create_model(sess, actions, batch_size)
    print("Model loaded")


    for key2d in test_set_2d.keys():

      (subj, b, fname) = key2d

      # choose SittingDown action to visualize
      if b  == 'SittingDown':
        print( "Subject: {}, action: {}, fname: {}".format(subj, b, fname) )

        # keys should be the same if 3d is in camera coordinates
        key3d = key2d if FLAGS.camera_frame else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
        key3d = (subj, b, fname[:-3]) if (fname.endswith('-sh')) and FLAGS.camera_frame else key3d

        enc_in  = test_set_2d[ key2d ]
        n2d, _ = enc_in.shape
        dec_out = test_set_3d[ key3d ]
        n3d, _ = dec_out.shape
        assert n2d == n3d

        # Split into about-same-size batches

        enc_in   = np.array_split( enc_in,  n2d // batch_size )
        dec_out  = np.array_split( dec_out, n3d // batch_size )

        # store all pose hypotheses in a list
        pose_3d_mdm = [[], [], [], [], []]

        for bidx in range( len(enc_in) ):

          # Dropout probability 0 (keep probability 1) for sampling
          dp = 1.0
          loss, _, out_all_components = model.step(sess, enc_in[bidx], dec_out[bidx], dp, isTraining=False)

          # denormalize the input 2d pose, ground truth 3d pose as well as 3d pose hypotheses from mdm
          out_all_components = np.reshape(out_all_components, [-1, model.HUMAN_3D_SIZE + 2, model.num_models])
          out_mean = out_all_components[:, : model.HUMAN_3D_SIZE, :]

          enc_in[bidx]  = data_utils.unNormalizeData(  enc_in[bidx], data_mean_2d, data_std_2d, dim_to_ignore_2d )
          dec_out[bidx] = data_utils.unNormalizeData( dec_out[bidx], data_mean_3d, data_std_3d, dim_to_ignore_3d )
          poses3d = np.zeros((out_mean.shape[0], 96, out_mean.shape[-1]))
          for j in range(out_mean.shape[-1]):
            poses3d[:, :, j] = data_utils.unNormalizeData( out_mean[:, :, j], data_mean_3d, data_std_3d, dim_to_ignore_3d )

          # extract the 17 joints
          dtu3d = np.hstack((np.arange(3), dim_to_use_3d)) if not (FLAGS.predict_14) else  dim_to_use_3d
          dec_out_17 = dec_out[bidx][: , dtu3d]
          pose_3d_17 = poses3d[:, dtu3d, :]
          sqerr = (pose_3d_17 - np.expand_dims(dec_out_17, axis=2)) ** 2
          dists = np.zeros((sqerr.shape[0], n_joints, sqerr.shape[2]))
          for m in range(dists.shape[-1]):
            dist_idx = 0
            for k in np.arange(0, n_joints * 3, 3):
              dists[:, dist_idx, m] = np.sqrt(np.sum(sqerr[:, k:k + 3, m], axis=1))
              dist_idx = dist_idx + 1

          [pose_3d_mdm[i].append(poses3d[:, :, i]) for i in range(poses3d.shape[-1])]

        # Put all the poses together
        enc_in, dec_out= map(np.vstack,[enc_in, dec_out])
        for i in range(poses3d.shape[-1]):
          pose_3d_mdm[i] = np.vstack(pose_3d_mdm[i])

          # Convert back to world coordinates
        if FLAGS.camera_frame:
          N_CAMERAS = 4
          N_JOINTS_H36M = 32

          # Add global position back
          dec_out = dec_out + np.tile( test_root_positions[ key3d ], [1,N_JOINTS_H36M] )
          for i in range(poses3d.shape[-1]):
            pose_3d_mdm[i] = pose_3d_mdm[i] + np.tile(test_root_positions[key3d], [1, N_JOINTS_H36M])


          # Load the appropriate camera
          subj, action, sname = key3d

          cname = sname.split('.')[1] # <-- camera name
          scams = {(subj,c+1): rcams[(subj,c+1)] for c in range(N_CAMERAS)} # cams of this subject
          scam_idx = [scams[(subj,c+1)][-1] for c in range(N_CAMERAS)].index( cname ) # index of camera used
          the_cam  = scams[(subj, scam_idx+1)] # <-- the camera used
          R, T, f, c, k, p, name = the_cam
          assert name == cname

          def cam2world_centered(data_3d_camframe):
            data_3d_worldframe = cameras.camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
            data_3d_worldframe = data_3d_worldframe.reshape((-1, N_JOINTS_H36M*3))
            # subtract root translation
            return data_3d_worldframe - np.tile( data_3d_worldframe[:,:3], (1,N_JOINTS_H36M) )

          # Apply inverse rotation and translation
          dec_out = cam2world_centered(dec_out)
          for i in range(poses3d.shape[-1]):
            pose_3d_mdm[i] = cam2world_centered(pose_3d_mdm[i])

        # sample some results to visualize
        np.random.seed(42)
        idx = np.random.permutation(enc_in.shape[0])
        enc_in, dec_out = enc_in[idx, :], dec_out[idx,:]
        for i in range(poses3d.shape[-1]):
          pose_3d_mdm[i] = pose_3d_mdm[i][idx, :]

        exidx = 1
        nsamples = 20

        for i in np.arange(nsamples):
          fig = plt.figure(figsize=(20, 5))

          subplot_idx = 1
          gs1 = gridspec.GridSpec(1, 7)  # 5 rows, 9 columns
          gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
          plt.axis('off')

          # Plot 2d pose
          ax1 = plt.subplot(gs1[subplot_idx - 1])
          p2d = enc_in[exidx, :]
          viz.show2Dpose(p2d, ax1)
          ax1.invert_yaxis()

          # Plot 3d gt
          ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
          p3d = dec_out[exidx, :]
          viz.show3Dpose(p3d, ax2)

          # Plot 3d pose hypotheses

          for i in range(poses3d.shape[-1]):
            ax3 = plt.subplot(gs1[subplot_idx + i + 1], projection='3d')
            p3d = pose_3d_mdm[i][exidx]
            viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71")
          # plt.show()
          plt.savefig('{}/sample_{}_{}_{}_{}.png'.format(path, subj, action, scam_idx, exidx))
          plt.close(fig)
          exidx = exidx + 1


def main(_):
  if FLAGS.sample:
    sample()
  elif FLAGS.test:
    test()
  else:
    train()

if __name__ == "__main__":

  tf.app.run()
