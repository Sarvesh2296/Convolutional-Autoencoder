import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.slim as slim
import utils_preprocess as utils
import os

latent_dim = 21
intermediate_dim = 1024

def lrelu(x, leak=0.2, name='lrelu'):
	with tf.variable_scope(name):
		f1 = 0.5*(1+leak)
		f2 = 0.5*(1-leak)
		return f1*x + f2*abs(x)


def encoder(image):
	print('Encoder')
	print(image.shape)
	input = slim.convolution2d(image, 3, [3,3],
	                           stride=[1,1],
	                           padding='SAME',
		                   weights_initializer = intitializer,
		                   activation_fn = lrelu,
		                   scope = 'en_conv1')
	print(input.shape)
	net = slim.convolution2d(input, 64, [3,3],
				stride=[2,2],
				padding='SAME',
				weights_initializer = intitializer,
				activation_fn = lrelu,
				scope = 'en_conv2')
	print(net.shape)
	net = slim.convolution2d(net, 128, [3,3],
				stride = [2,2],
				padding = 'SAME',
				weights_initializer = intitializer,
				activation_fn = lrelu,
				scope = 'en_conv3')
	print(net.shape)
	net = slim.convolution2d(net, 128, [3,3],
				stride = [1,1],
				padding = 'SAME',
				weights_initializer = intitializer,
				activation_fn = lrelu,
				scope = 'en_conv4')
	print(net.shape)
	net = slim.fully_connected(slim.flatten(net), intermediate_dim,
				weights_initializer = intitializer,
				activation_fn = lrelu,
				scope = 'en_flat1')
	print(net.shape)
	latent = slim.fully_connected(net, latent_dim,
				weights_initializer = intitializer,
				activation_fn = lrelu,
				scope = 'en_latent')
	print(latent.shape)
	return latent

def decoder(z):
	print('Decoder')
	input = slim.fully_connected(z, intermediate_dim,
				weights_initializer = intitializer,
				activation_fn = lrelu,
				scope = 'de_intermediate')
	print(input.shape)
	net = slim.fully_connected(input, 64*64*128,
				weights_initializer = intitializer,
				activation_fn = lrelu,
				scope = 'de_flat2')
	print(net.shape)
	net = tf.reshape(net, [1, 64, 64, 128])
	print(net.shape)
	net = slim.convolution2d_transpose(net, 128, [3,3],
				stride = [1,1],
				weights_initializer = intitializer,
				activation_fn = lrelu,
				scope = 'de_deconv1')
	print(net.shape)
	net = slim.convolution2d_transpose(net, 64, [3,3],
				stride = [2,2],
				weights_initializer = intitializer,
				activation_fn = lrelu,
				scope = 'de_deconv2')
	print(net.shape)
	output = slim.convolution2d_transpose(net, 3, [3,3],
				stride = [2,2],
				weights_initializer = intitializer,
				activation_fn = tf.tanh,
				scope = 'de_output')
	print(output.shape)
	return output

g = tf.Graph()

intitializer = tf.truncated_normal_initializer(stddev = 0.02)
#image = tf.placeholder(tf.float32, [1, 256, 256, 3])
image = utils.load_image()
z = encoder(image)
output = decoder(z)

# L2 reg not required https://pgaleone.eu/neural-networks/deep-learning/2016/12/13/convolutional-autoencoders-in-tensorflow/
reconst_loss = slim.losses.mean_squared_error(image, output)
optimizer = tf.train.AdamOptimizer(0.01)
train_op = slim.learning.create_train_op(reconst_loss, optimizer)
tf.summary.scalar('losses', reconst_loss)

checkpoints_dir = "./Summary"
if not os.path.exists(checkpoints_dir):
	os.mkdir(checkpoints_dir)

slim.learning.train(train_op,
		checkpoints_dir,
		save_summaries_secs = 60,
		save_interval_secs = 60)


	
