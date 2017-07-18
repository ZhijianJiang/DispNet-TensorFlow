# -*- coding:utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import time
from datetime import date
import cPickle

from six.moves import urllib
from PIL import Image
import tensorflow as tf
import numpy as np
#from tensorflow.core.protobuf import saver_pb2
import matplotlib.pyplot as plt

IMAGE_SIZE_X = 768
IMAGE_SIZE_Y = 384
BATCH_SIZE = 12
ROUND_STEP = 12
TRAINING_ROUNDS = 50
LEARNING_RATE =1e-5
#MODELS_PATH = "/home/visg1/jzj/models/model.ckpt.data-00000-of-00001"
MODELS_DIR = '/home/visg1/jzj/my-model'
DATA_DIR = '/home/visg1/jzj/Data_fin'
LOGS_DIR = '/home/visg1/jzj/logs'
RUNNING_LOGS_DIR = '/home/visg1/jzj/running_logs'
OUTPUT_DIR = '/home/visg1/jzj/output'
GT_DIR = '/home/visg1/jzj/gopro.pkl'
TRAIN_SERIES = range(132) + range(136, 140) + range(144, 150) + range(160, 198) + range(202, 220) + range(224, 230) + range(234, 248) + range(258, 292) + range(296, 302) + range(306, 324)
image_num = np.size(TRAIN_SERIES)


def py_avg_pool(value, strides):
	batch_size, height, width, channel_size = value.shape
	res_height = int(height / strides[1])
	res_width = int(width / strides[2])
	print(res_height, res_width)
	result = np.zeros((batch_size, res_height, res_width, 1))
	for i in range(res_height):
		for j in range(res_width):
			for k in range(batch_size):
				result[k, i, j, 0] = np.mean(value[k, i * int(strides[1]) : (i + 1) * int(strides[1]), j * int(strides[2]) : (j + 1) * int(strides[2]), :])
	return result



def weight_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.Variable(initializer(shape=shape), name='weight')
    #return tf.Variable(initial, name='W')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='b')

def conv2d(x, W, strides):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def upconv2d_2x2(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, filter=W, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME');

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool')

def loss(pre, gt):
    loss = tf.sqrt(tf.reduce_mean(tf.square(pre - gt)))
    return loss

def pre(conv):
    return tf.expand_dims(tf.reduce_mean(conv, 3), -1)
    #return tf.reduce_mean(conv, 3)

def model(combine_image, ground_truth):
  # conv1
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([7,7, 6,64]) 
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(combine_image, W_conv1, [1, 2, 2 ,1]) + b_conv1) 
    #h_pool1 = max_pool_2x2(h_conv1)   
    # h_pool1 = h_conv1 

  # conv2
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5,5, 64,128]) 
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, [ 1, 2, 2, 1]) + b_conv2) 
    #h_pool2 = max_pool_2x2(h_conv2)   
    h_pool2 = h_conv2                                      
  # conv3a
  with tf.name_scope('conv3a'):
    W_conv3a = weight_variable([5,5, 128,256]) 
    b_conv3a = bias_variable([256])
    h_conv3a = tf.nn.relu(conv2d(h_conv2, W_conv3a, [1, 2, 2, 1]) + b_conv3a) 
    #h_pool3a = max_pool_2x2(h_conv3a)
    #h_pool3a = h_conv3a

  # conv3b
  with tf.name_scope('conv3b'):
    W_conv3b = weight_variable([3,3, 256,256]) 
    b_conv3b = bias_variable([256])
    h_conv3b = tf.nn.relu(conv2d(h_conv3a, W_conv3b, [1, 1, 1, 1]) + b_conv3b) 
    #h_pool3b = h_conv3b

  # conv4a
  with tf.name_scope('conv4a'):
    W_conv4a = weight_variable([3,3, 256,512]) 
    b_conv4a = bias_variable([512])
    h_conv4a = tf.nn.relu(conv2d(h_conv3b, W_conv4a, [1, 2, 2, 1]) + b_conv4a) 
    #h_pool4a = max_pool_2x2(h_conv4a) 
    #h_pool4a = h_conv4a
  # conv4b
  with tf.name_scope('conv4b'):
    W_conv4b = weight_variable([3,3, 512,512]) 
    b_conv4b = bias_variable([512])
    h_conv4b = tf.nn.relu(conv2d(h_conv4a, W_conv4b, [1, 1, 1, 1]) + b_conv4b) 
    #h_pool4b = h_conv4b

  # conv5a
  with tf.name_scope('conv5a'):
    W_conv5a = weight_variable([3,3, 512,512]) 
    b_conv5a = bias_variable([512])
    h_conv5a = tf.nn.relu(conv2d(h_conv4b, W_conv5a, [1, 2, 2, 1]) + b_conv5a) 
    #h_pool5a = max_pool_2x2(h_conv5a) 
    #h_pool5a = h_conv5a
  # conv5b
  with tf.name_scope('conv5b'):
    W_conv5b = weight_variable([3,3, 512,512]) 
    b_conv5b = bias_variable([512])
    h_conv5b = tf.nn.relu(conv2d(h_conv5a, W_conv5b, [ 1, 1, 1, 1]) + b_conv5b) 
    #h_pool5b = h_conv5b

  # conv6a
  with tf.name_scope('conv6a'):
    W_conv6a = weight_variable([3,3, 512,1024]) 
    b_conv6a = bias_variable([1024])
    h_conv6a = tf.nn.relu(conv2d(h_conv5b, W_conv6a, [1, 2, 2, 1]) + b_conv6a) 
    #h_pool6a = max_pool_2x2(h_conv6a) 
    #h_pool6a = h_conv6a
  # conv6b
  with tf.name_scope('conv6b'):
    W_conv6b = weight_variable([3,3, 1024,1024]) 
    b_conv6b = bias_variable([1024])
    h_conv6b = tf.nn.relu(conv2d(h_conv6a, W_conv6b, [1, 1, 1, 1]) + b_conv6b) 
    #h_pool6b = h_conv6b

  # pr6 + loss6
  with tf.name_scope('pr6_loss6'):
    W_pr6 = weight_variable([3,3, 1024,1]) 
    b_pr6 = bias_variable([1])
    pr6 = tf.nn.relu(conv2d(h_conv6b, W_pr6, [1, 1, 1, 1]) + b_pr6)
    # h_conv6b = tf.nn.relu(conv2d(h_conv6a, W_conv6b, [1, 1, 1, 1]) + b_conv6b)
    # pr6 = pre(h_conv6b)
    gt6 = tf.nn.avg_pool(ground_truth, ksize=[1,64,64,1], strides=[1,64,64,1], padding='SAME', name='gt6')
    loss6 = loss(pr6, gt6)

  # upconv5
  with tf.name_scope('upconv5'):
    W_upconv5 = weight_variable([4,4, 512,1024]) 
    b_upconv5 = bias_variable([512])
    h_upconv5 = tf.nn.relu(tf.contrib.layers.batch_norm(upconv2d_2x2(h_conv6b,  W_upconv5, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 32), np.int32(IMAGE_SIZE_X / 32), 512]) + b_upconv5, center=True, scale=True, is_training=True)) 


  # iconv5
  with tf.name_scope('iconv5'):
    W_iconv5 = weight_variable([3,3, 1024,512]) 
    b_iconv5 = bias_variable([512])
    h_iconv5 = tf.nn.relu(conv2d(tf.concat([h_upconv5, h_conv5b], 3), W_iconv5, [1, 1, 1, 1]) + b_iconv5) 


  # pr5 + loss5
  with tf.name_scope('pr5_loss5'):
    W_pr5 = weight_variable([3,3, 512,1]) 
    b_pr5 = bias_variable([1])
    pr5 = tf.nn.relu(conv2d(h_iconv5, W_pr5, [1, 1, 1, 1]) + b_pr5)    
    # pr5 = pre(h_iconv5)
    gt5 = tf.nn.avg_pool(ground_truth, ksize=[1,32,32,1], strides=[1,32,32,1], padding='SAME', name='gt5')
    loss5 = loss(pr5, gt5)

  # upconv4
  with tf.name_scope('upconv4'):
    W_upconv4 = weight_variable([4,4, 256, 512])
    b_upconv4 = bias_variable([256])
    h_upconv4 = tf.nn.relu(tf.contrib.layers.batch_norm(upconv2d_2x2(h_iconv5, W_upconv4, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 16), np.int32(IMAGE_SIZE_X / 16), 256]) + b_upconv4, center=True, scale=True, is_training=True))


  # iconv4
  with tf.name_scope('iconv4'):
    W_iconv4 = weight_variable([3,3, 768,256]) 
    b_iconv4 = bias_variable([256])
    h_iconv4 = tf.nn.relu(conv2d(tf.concat([h_upconv4, h_conv4b], 3), W_iconv4, [ 1, 1, 1, 1]) + b_iconv4) 


  # pr4 + loss4
  with tf.name_scope('pr4_loss4'):
    W_pr4 = weight_variable([3,3, 256,1]) 
    b_pr4 = bias_variable([1])
    pr4 = tf.nn.relu(conv2d(h_iconv4, W_pr4, [1, 1, 1, 1]) + b_pr4)    
    # pr4 = pre(h_iconv4)
    gt4 = tf.nn.avg_pool(ground_truth, ksize=[1,16,16,1], strides=[1,16,16,1], padding='SAME', name='gt4')
    loss4 = loss(pr4, gt4)

  # upconv3
  with tf.name_scope('upconv3'):
    W_upconv3 = weight_variable([4,4,128, 256]) 
    b_upconv3 = bias_variable([128])
    h_upconv3 = tf.nn.relu(tf.contrib.layers.batch_norm(upconv2d_2x2(h_iconv4, W_upconv3, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 8), np.int32(IMAGE_SIZE_X / 8), 128]) + b_upconv3, center=True, scale=True, is_training=True)) 


  # iconv3
  with tf.name_scope('iconv3'):
    W_iconv3 = weight_variable([3,3, 384,128]) 
    b_iconv3 = bias_variable([128])
    h_iconv3 = tf.nn.relu(conv2d(tf.concat([h_upconv3, h_conv3b], 3), W_iconv3, [ 1, 1, 1, 1]) + b_iconv3) 


  # pr3 + loss3
  with tf.name_scope('pr3_loss3'):
    W_pr3 = weight_variable([3,3, 128,1]) 
    b_pr3 = bias_variable([1])
    pr3 = tf.nn.relu(conv2d(h_iconv3, W_pr3, [1, 1, 1, 1]) + b_pr3) 
    # pr3 = pre(h_iconv3)
    gt3 = tf.nn.avg_pool(ground_truth, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME', name='gt')
    loss3 = loss(pr3, gt3)

  # upconv2
  with tf.name_scope('upconv2'):
    W_upconv2 = weight_variable([4,4,64, 128]) 
    b_upconv2 = bias_variable([64])
    h_upconv2 = tf.nn.relu(tf.contrib.layers.batch_norm(upconv2d_2x2(h_iconv3, W_upconv2, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 4), np.int32(IMAGE_SIZE_X / 4), 64]) + b_upconv2, center=True, scale=True, is_training=True)) 


  # iconv2
  with tf.name_scope('iconv2'):
    W_iconv2 = weight_variable([3,3, 192,64]) 
    b_iconv2 = bias_variable([64])
    h_iconv2 = tf.nn.relu(conv2d(tf.concat([h_upconv2, h_conv2], 3), W_iconv2, [1, 1, 1, 1]) + b_iconv2) 


  # pr2 + loss2
  with tf.name_scope('pr2_loss2'):
    W_pr2 = weight_variable([3,3, 64,1]) 
    b_pr2 = bias_variable([1])
    pr2 = tf.nn.relu(conv2d(h_iconv2, W_pr2, [1, 1, 1, 1]) + b_pr2) 
    # pr2 = pre(h_iconv2)
    gt2 = tf.nn.avg_pool(ground_truth, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME', name='gt')
    loss2 = loss(pr2, gt2)

  # upconv1
  with tf.name_scope('upconv1'):
    W_upconv1 = weight_variable([4,4,32, 64]) 
    b_upconv1 = bias_variable([32])
    h_upconv1 = tf.nn.relu(tf.contrib.layers.batch_norm(upconv2d_2x2(h_iconv2, W_upconv1, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 2), np.int32(IMAGE_SIZE_X / 2), 32]) + b_upconv1, center=True, scale=True, is_training=True)) 

  # iconv1
  with tf.name_scope('iconv1'):
    W_iconv1 = weight_variable([3,3, 96,32]) 
    b_iconv1 = bias_variable([32])
    h_iconv1 = tf.nn.relu(conv2d(tf.concat([h_upconv1, h_conv1], 3), W_iconv1, [ 1, 1, 1, 1]) + b_iconv1) 

  # pr1 + loss1
  with tf.name_scope('pr1_loss1'):
    W_pr1 = weight_variable([3,3, 32,1]) 
    b_pr1 = bias_variable([1])
    pr1 = tf.nn.relu(conv2d(h_iconv1, W_pr1, [1, 1, 1, 1]) + b_pr1) 
    # pr1 = pre(h_iconv1)
    gt1 = tf.nn.avg_pool(ground_truth, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='gt')
    loss1 = loss(pr1, gt1)
  

  final_output = pr1
  # overall loss
  with tf.name_scope('loss'):
    total_loss = ( 1/2 * loss1 + 1/4 * loss2 + 1/8 * loss3 + 1/16 * loss4 + 1/32 * loss5 + 1/32 * loss6)
  return final_output, total_loss, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2, pr1, tf.is_inf(loss6)

def _norm(img):
  return (img - np.mean(img)) / np.std(img)

def main():
  with open(GT_DIR) as f:
    buf = cPickle.load(f)
  image_left = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3], name='image_left')
  image_right = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3], name='image_right')
  ground_truth = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1], name='ground_truth')
  combine_image = tf.concat([image_left, image_right], 3)
  final_output, total_loss, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2, pr1, loss6_inf = model(combine_image=combine_image, 
                            ground_truth=ground_truth)
  tf.summary.scalar('loss', total_loss)

  with tf.name_scope('train'):
      optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss)
 
  merged = tf.summary.merge_all()

  # important step
  sess = tf.Session()
  
  if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
      writer = tf.train.SummaryWriter(LOGS_DIR, sess.graph)
  else: # tensorflow version >= 0.12
      writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)


  left_images = sorted(os.listdir(DATA_DIR+ '/left/'))
  right_images = sorted(os.listdir(DATA_DIR + '/right/'))
  # output_images = sorted(os.listdir(DATA_DIR + '/output/'))  


  # tf.initialize_all_variables() no long valid from
  # 2017-03-02 if using tensorflow >= 0.12

  if int((tf.__version__).split('.')[1]) < 12:
      init = tf.initialize_all_variables()
  else:
      init = tf.global_variables_initializer()
  # saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)
  saver = tf.train.Saver()
  
  
  sess.run(init)
  # saver.restore(sess, MODEL_PATH)
  with open(RUNNING_LOGS_DIR + "/log" + date.isoformat(date.today()) + str(time.time()) + ".txt", "w+") as file:
    file.write('BATCH_SIZE ' + str(BATCH_SIZE) + '\n'
	+ ' TRAINING_ROUNDS ' + str(TRAINING_ROUNDS) + '\n'
	 + ' image_num ' + str(image_num) + '\n' 
	+ ' LEARNING_RATE ' + str(LEARNING_RATE) + '\n')

    for round in range(TRAINING_ROUNDS):
      for i in range(0 , image_num - BATCH_SIZE, ROUND_STEP):

        for j in range(BATCH_SIZE):
          # input data
          full_pic_name = DATA_DIR+ '/left/' + left_images[TRAIN_SERIES[i + j]]
          input_one_image = Image.open(full_pic_name)
          input_one_image = _norm(np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
          if(j == 0):
  	        input_left_images = input_one_image
          else:
            input_left_images = np.concatenate((input_left_images, input_one_image), axis=0)

          full_pic_name = DATA_DIR + '/right/' + right_images[TRAIN_SERIES[i + j]]
          input_one_image = Image.open(full_pic_name)
          input_one_image = _norm(np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
          if(j == 0):
  	        input_right_images = input_one_image
          else:
            input_right_images = np.concatenate((input_right_images, input_one_image), axis=0)

          input_one_image = buf[TRAIN_SERIES[i + j]]
          input_one_image = np.reshape(input_one_image, (IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
          input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))

          if(j == 0):
  	        input_gts = input_one_image
          else:
            input_gts = np.concatenate((input_gts, input_one_image), axis=0)
        result, optimizer_res, total_loss_res, loss1_res, loss2_res, loss3_res, loss4_res, loss5_res, loss6_res, pr6_res, pr5_res, pr4_res, pr3_res, pr2_res, pr1_res, loss6_inf_res =sess.run([merged, optimizer, total_loss, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2, pr1, loss6_inf],feed_dict={image_left:input_left_images, image_right:input_right_images, ground_truth:input_gts})
	      
        if round == TRAINING_ROUNDS - 1:
          final_result = (sess.run(final_output, feed_dict={image_left:input_left_images, image_right:input_right_images, ground_truth:input_gts})) 
          for k in range(1):
            result = np.squeeze(final_result[k])
            result = result.astype(np.uint8)
            plt.imsave(OUTPUT_DIR + '/' + str(i) + '.png', result, format='png')
            file.write('round ' + str(round) + ' batch ' + str(i) + ' total_loss ' + str(total_loss_res) +' loss1 ' + str(loss1_res) +  '\n')

        if i == 0:
			print(' pr1_real_loss ' + str(np.sqrt(np.mean(np.square(py_avg_pool(input_gts, [1,2,2,1]), pr1_res)))))
			file.write('round ' + str(round) + ' batch ' + str(i) + ' total_loss ' + str(total_loss_res) +' loss1 ' + str(loss1_res) +  '\n')

if __name__ == '__main__':
  main()
