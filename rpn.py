import os;
import math;
import numpy as np;
import tensorflow as tf;
from nn import *;

class RPN(object):
	def __init__(self,params,total_num_anchor):
		self.params = params;
		self.cnn_model = params.cnn;
		self.batch_size = params.batch_size;
		self.batch_norm = params.batch_norm;
		self.total_num_anchor = total_num_anchor;
		if(self.cnn_model=='vgg16'):
			self.feature_shape = [40,40,512];
		else:
			self.feature_shape = [20,20,2048];
		self.build();

	def build(self):
		bn = self.batch_norm;
		train = tf.placeholder(tf.bool);
		cnn_features = tf.placeholder(tf.float32, [self.batch_size]+self.feature_shape);
		anchor_labels = tf.placeholder(tf.int32, [self.batch_size, self.total_num_anchor]);
		anchor_regs = tf.placeholder(tf.float32, [self.batch_size, self.total_num_anchor, 4]);
		anchor_masks = tf.placeholder(tf.float32, [self.batch_size, self.total_num_anchor]);
		anchor_weights = tf.placeholder(tf.float32, [self.batch_size, self.total_num_anchor]);
		anchor_reg_masks = tf.placeholder(tf.float32, [self.batch_size, self.total_num_anchor]);
		rpn_logits = [];
		rpn_regs = [];
		features = cnn_features;
		if self.cnn_model == 'vgg16':
			kernel_size = [10,10];
		else:
			kernel_size = [5,5];

		for i in range(2):
			rpn1 = convolution(features,kernel_size[0],kernel_size[1],512,1,1,'rpn1_'+str(i),group_id=1);
			rpn1 = nonlinear(rpn1,'relu');
			rpn1 = dropout(rpn1,0.5,train);
			for j in range(9):
				logits = convolution(rpn1,1,1,2,1,1,'rpn_logits'+str(i)+'_'+str(j),group_id=1);
				logits = tf.reshape(logits,[self.batch_size,-1,2]);
				rpn_logits.append(logits);
				regs = convolution(rpn1,1,1,4,1,1,'rpn_regs'+str(i)+'_'+str(j),group_id=1);
				regs = tf.clip_by_value(regs,-0.2,0.2);
				regs = tf.reshape(regs,[self.batch_size,-1,4]);
				rpn_regs.append(regs);
			if i<1:
				features = max_pool(features,2,2,2,2,'rpn_pool_'+str(i));

		rpn_logits = tf.concat(rpn_logits,1);
		rpn_regs = tf.concat(rpn_regs,1);
		rpn_logits = tf.reshape(rpn_logits,[-1,2]);
		rpn_regs = tf.reshape(rpn_regs,[-1,4]);
		anchor_labels = tf.reshape(anchor_labels,[-1]);       
		anchor_regs = tf.reshape(anchor_regs,[-1, 4]);
		anchor_masks = tf.reshape(anchor_masks,[-1]);
		anchor_weights = tf.reshape(anchor_weights,[-1]);
		anchor_reg_masks = tf.reshape(anchor_reg_masks,[-1]);

		loss0 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_logits,labels=anchor_labels)*anchor_masks;
		loss0 = tf.reduce_sum(loss0*anchor_weights)/tf.reduce_sum(anchor_weights);
		reg_loss = l2_loss(rpn_regs,anchor_regs)*anchor_reg_masks;
		reg_avg = tf.reduce_sum(anchor_reg_masks);
		loss0 = tf.cond(tf.less(0.0,reg_avg), lambda: loss0+self.params.rpn_relative*tf.reduce_sum(reg_loss)/reg_avg,lambda: loss0);
		loss1 = self.params.weight_decay*tf.add_n(tf.get_collection('l2_1'));
		loss = loss0+loss1;

		rpn_probabilities = tf.nn.softmax(rpn_logits);
		rpn_scores = tf.squeeze(tf.slice(rpn_probabilities,[0,1],[-1,1]));
		rpn_scores = tf.reshape(rpn_scores,[self.batch_size,self.total_num_anchor]);
		rpn_regs = tf.reshape(rpn_regs,[self.batch_size,self.total_num_anchor,4]);                     

		self.train = train;
		self.features = cnn_features;
		self.anchor_labels = anchor_labels
		self.anchor_regs = anchor_regs;
		self.anchor_masks = anchor_masks;
		self.anchor_weights = anchor_weights;
		self.anchor_reg_masks = anchor_reg_masks;
		self.rpn_loss = loss;
		self.rpn_loss0 = loss0;
		self.rpn_loss1 = loss1;
		self.rpn_scores = rpn_scores;
		self.rpn_regs = rpn_regs;
		print("RPN built......");