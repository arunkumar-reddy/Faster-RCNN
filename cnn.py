import os;
import math;
import numpy as np;
import tensorflow as tf;
from nn import *;

class CNN(object):
	def __init__(self,params):
		self.cnn_model = params.cnn;
		self.batch_size = params.batch_size;
		self.batch_norm = params.batch_norm;
		if(self.cnn_model=='vgg16'):
			self.VGG16();
		elif(self.cnn_model=='resnet50'):
			self.ResNet50();
		elif(self.cnn_model=='resnet101'):
			self.ResNet101();
		elif(self.cnn_model=='resnet152'):
			self.ResNet152();

	def VGG16(self):
		print('Building the VGG-16 component......');
		image_shape = [640,640,3];
		bn = self.batch_norm;
		images = tf.placeholder(tf.float32,[self.batch_size]+image_shape);
		train = tf.placeholder(tf.bool);

		conv1_1 = convolution(images,3,3,64,1,1,'conv1_1');
		conv1_1 = batch_norm(conv1_1,'bn1_1',train,bn,'relu');
		conv1_2 = convolution(conv1_1,3,3,64,1,1,'conv1_2');
		conv1_2 = batch_norm(conv1_2,'bn1_2',train,bn,'relu');
		pool1 = max_pool(conv1_2,2,2,2,2,'pool1');

		conv2_1 = convolution(pool1,3,3,128,1,1,'conv2_1');
		conv2_1 = batch_norm(conv2_1,'bn2_1',train,bn,'relu');
		conv2_2 = convolution(conv2_1,3,3,128,1,1,'conv2_2');
		conv2_2 = batch_norm(conv2_2,'bn2_2',train,bn,'relu');
		pool2 = max_pool(conv2_2,2,2,2,2,'pool2');

		conv3_1 = convolution(pool2,3,3,256,1,1,'conv3_1');
		conv3_1 = batch_norm(conv3_1,'bn3_1',train,bn,'relu');
		conv3_2 = convolution(conv3_1,3,3,256,1,1,'conv3_2');
		conv3_2 = batch_norm(conv3_2,'bn3_2',train,bn,'relu');
		conv3_3 = convolution(conv3_2,3,3,256,1,1,'conv3_3');
		conv3_3 = batch_norm(conv3_3,'bn3_3',train,bn,'relu');
		pool3 = max_pool(conv3_3,2,2,2,2,'pool3');

		conv4_1 = convolution(pool3,3,3,512,1,1,'conv4_1');
		conv4_1 = batch_norm(conv4_1,'bn4_1',train,bn,'relu');
		conv4_2 = convolution(conv4_1,3,3,512,1,1,'conv4_2');
		conv4_2 = batch_norm(conv4_2,'bn4_2',train,bn,'relu');
		conv4_3 = convolution(conv4_2,3,3,512,1,1,'conv4_3');
		conv4_3 = batch_norm(conv4_3,'bn4_3',train,bn,'relu');
		pool4 = max_pool(conv4_3,2,2,2,2,'pool4');

		conv5_1 = convolution(pool4,3,3,512,1,1,'conv5_1');
		conv5_1 = batch_norm(conv5_1,'bn5_1',train,bn,'relu');
		conv5_2 = convolution(conv5_1,3,3,512,1,1,'conv5_2');
		conv5_2 = batch_norm(conv5_2,'bn5_2',train,bn,'relu');
		conv5_3 = convolution(conv5_2,3,3,512,1,1,'conv5_3')
		conv5_3 = batch_norm(conv5_3,'bn5_3',train,bn,'relu');

		self.features = conv5_3;
		self.feature_shape = [40,40,512];
		self.images = images;
		self.train = train;
		print('VGG-16 built......');

	def basic_block1(self,feats,name1,name2,train,bn,c,s=2):
		'''A basic block of ResNets'''
		branch1 = convolution_no_bias(feats,1,1,4*c,s,s,name1+'_branch1');
		branch1 = batch_norm(branch1,name2+'_branch1',train,bn,None);
		branch2a = convolution_no_bias(feats,1,1,c,s,s,name1+'_branch2a');
		branch2a = batch_norm(branch2a,name2+'_branch2a',train,bn,'relu');
		branch2b = convolution_no_bias(branch2a,3,3,c,1,1,name1+'_branch2b');
		branch2b = batch_norm(branch2b,name2+'_branch2b',train,bn,'relu');
		branch2c = convolution_no_bias(branch2b,1,1,4*c,1,1,name1+'_branch2c');
		branch2c = batch_norm(branch2c,name2+'_branch2c',train,bn,None);
		output = branch1+branch2c;
		output = nonlinear(output, 'relu');
		return output;

	def basic_block2(self,feats,name1,name2,train,bn,c):
		'''Another basic block of ResNets'''
		branch2a = convolution_no_bias(feats,1,1,c,1,1,name1+'_branch2a');
		branch2a = batch_norm(branch2a,name2+'_branch2a',train,bn,'relu');
		branch2b = convolution_no_bias(branch2a,3,3,c,1,1,name1+'_branch2b');
		branch2b = batch_norm(branch2b,name2+'_branch2b',train,bn,'relu');
		branch2c = convolution_no_bias(branch2b,1,1,4*c,1,1,name1+'_branch2c');
		branch2c = batch_norm(branch2c,name2+'_branch2c',train,bn,None);
		output = feats+branch2c;
		output = nonlinear(output,'relu');
		return output;

	def ResNet50(self):
		print('Building the ResNet50 component......');
		image_shape = [640,480,3];
		bn = self.batch_norm;
		images = tf.placeholder(tf.float32,[self.batch_size]+image_shape);
		train = tf.placeholder(tf.bool);

		conv1 = convolution(images,7,7,64,2,2,'conv1');
		conv1 = batch_norm(conv1,'bn_conv1',train,bn,'relu');
		pool1 = max_pool(conv1,3,3,2,2,'pool1');

		res2a = self.basic_block1(pool1,'res2a','bn2a',train,bn,64,1);
		res2b = self.basic_block2(res2a,'res2b','bn2b',train,bn,64);
		res2c = self.basic_block2(res2b,'res2c','bn2c',train,bn,64);
  
		res3a = self.basic_block1(res2c,'res3a','bn3a',train,bn,128);
		res3b = self.basic_block2(res3a,'res3b','bn3b',train,bn,128);
		res3c = self.basic_block2(res3b,'res3c','bn3c',train,bn,128);
		res3d = self.basic_block2(res3c,'res3d','bn3d',train,bn,128);

		res4a = self.basic_block1(res3d,'res4a','bn4a',train,bn,256);
		res4b = self.basic_block2(res4a,'res4b','bn4b',train,bn,256);
		res4c = self.basic_block2(res4b,'res4c','bn4c',train,bn,256);
		res4d = self.basic_block2(res4c,'res4d','bn4d',train,bn,256);
		res4e = self.basic_block2(res4d,'res4e','bn4e',train,bn,256);
		res4f = self.basic_block2(res4e,'res4f','bn4f',train,bn,256);

		res5a = self.basic_block1(res4f,'res5a','bn5a',train,bn,512);
		res5b = self.basic_block2(res5a,'res5b','bn5b',train,bn,512);
		res5c = self.basic_block2(res5b,'res5c','bn5c',train,bn,512);

		res5c = tf.reshape(res5c,[self.batch_size,49,2048]);
		self.features = res5c;
		self.feature_shape = [20,20,2048];
		self.images = images;
		self.train = train;
		print('ResNet50 built......');

	def ResNet101(self):
		print('Building the ResNet101 component......');
		image_shape = [640,480,3];
		bn = self.batch_norm;
		images = tf.placeholder(tf.float32,[self.batch_size]+image_shape);
		train = tf.placeholder(tf.bool);

		conv1 = convolution(images,7,7,64,2,2,'conv1');
		conv1 = batch_norm(conv1,'bn_conv1',train,bn,'relu');
		pool1 = max_pool(conv1,3,3,2,2,'pool1');

		res2a = self.basic_block1(pool1,'res2a','bn2a',train,bn,64,1);
		res2b = self.basic_block2(res2a,'res2b','bn2b',train,bn,64);
		res2c = self.basic_block2(res2b,'res2c','bn2c',train,bn,64);
  
		res3a = self.basic_block1(res2c,'res3a','bn3a',train,bn,128);       
		temp = res3a;
		for i in range(1,4):
			temp = self.basic_block2(temp,'res3b'+str(i),'bn3b'+str(i),train,bn,128);
		res3b3 = temp;
 
		res4a = self.basic_block1(res3b3,'res4a','bn4a',train,bn,256);
		temp = res4a;
		for i in range(1,23):
			temp = self.basic_block2(temp,'res4b'+str(i),'bn4b'+str(i),train,bn,256);
		res4b22 = temp;

		res5a = self.basic_block1(res4b22,'res5a','bn5a',train,bn,512);
		res5b = self.basic_block2(res5a,'res5b','bn5b',train,bn,512);
		res5c = self.basic_block2(res5b,'res5c','bn5c',train,bn,512);

		res5c = tf.reshape(res5c,[self.batch_size,49,2048]);
		self.features = res5c;
		self.feature_shape = [20,20,2048];
		self.images = images;
		self.train = train;
		print('ResNet101 built......');

	def ResNet152(self):
		print('Building the ResNet152 component......');
		image_shape = [640,480,3];
		bn = self.batch_norm;
		images = tf.placeholder(tf.float32,[self.batch_size]+image_shape);
		train = tf.placeholder(tf.bool);

		conv1 = convolution(images,7,7,64,2,2,'conv1');
		conv1 = batch_norm(conv1,'bn_conv1',train,bn,'relu');
		pool1 = max_pool(conv1,3,3,2,2,'pool1');

		res2a = self.basic_block1(pool1,'res2a','bn2a',train,bn,64,1);
		res2b = self.basic_block2(res2a,'res2b','bn2b',train,bn,64);
		res2c = self.basic_block2(res2b,'res2c','bn2c',train,bn,64);

		res3a = self.basic_block1(res2c,'res3a','bn3a',train,bn,128);       
		temp = res3a;
		for i in range(1,8):
			temp = self.basic_block2(temp,'res3b'+str(i),'bn3b'+str(i),train,bn,128);
		res3b7 = temp;
 
		res4a = self.basic_block1(res3b7,'res4a','bn4a',train,bn,256);
		temp = res4a;
		for i in range(1,36):
			temp = self.basic_block2(temp,'res4b'+str(i),'bn4b'+str(i),train,bn,256);
		res4b35 = temp;

		res5a = self.basic_block1(res4b35,'res5a','bn5a',train,bn,512);
		res5b = self.basic_block2(res5a,'res5b','bn5b',train,bn,512);
		res5c = self.basic_block2(res5b,'res5c','bn5c',train,bn,512);

		res5c = tf.reshape(res5c,[self.batch_size,49,2048]);
		self.features = res5c;
		self.feature_shape = [20,20,2048];
		self.images = images;
		self.train = train;
		print('ResNet152 built......');