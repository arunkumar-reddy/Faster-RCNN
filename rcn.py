import os;
import math;
import numpy as np;
import tensorflow as tf;
from nn import *;

class RCN(object):
	def __init__(self,params,roi_feature_shape,num_classes):
		self.params = params;
		self.cnn_model = params.cnn;
		self.batch_size = params.batch_size;
		self.batch_norm = params.batch_norm;
		self.roi_feature_shape = roi_feature_shape;
		self.num_classes = num_classes;
		self.build();

	def build(self):
		roi = self.params.roi;
		bn = self.batch_norm;
		train = tf.placeholder(tf.bool);
		roi_features = tf.placeholder(tf.float32,[self.batch_size,roi]+self.roi_feature_shape);  
		roi_classes = tf.placeholder(tf.int32,[self.batch_size,roi]); 
		roi_regs = tf.placeholder(tf.float32,[self.batch_size,roi,4]); 
		roi_masks = tf.placeholder(tf.float32,[self.batch_size,roi]); 
		roi_weights = tf.placeholder(tf.float32,[self.batch_size,roi]); 
		roi_reg_masks = tf.placeholder(tf.float32,[self.batch_size,roi]); 

		roi_features = tf.reshape(roi_features,[self.batch_size*roi]+self.roi_feature_shape);
		roi_pool_features = max_pool(roi_features,2,2,2,2,'roi_pool');
		roi_pool_features = tf.reshape(roi_pool_features,[self.batch_size*roi,-1]);

		fc6_feats = fully_connected(roi_pool_features,4096,'rcn_fc6',group_id=2);
		fc6_feats = nonlinear(fc6_feats,'relu');
		fc6_feats = dropout(fc6_feats,0.5,train);
		fc7_feats = fully_connected(fc6_feats,4096,'rcn_fc7',group_id=2);
		fc7_feats = nonlinear(fc7_feats,'relu');
		fc7_feats = dropout(fc7_feats,0.5,train);
		
		rcn_logits = fully_connected(fc7_feats,self.num_classes,'rcn_logits',group_id=2);
		roi_classes = tf.reshape(roi_classes,[-1]);
		roi_regs = tf.reshape(roi_regs,[-1, 4]);
		roi_masks = tf.reshape(roi_masks,[-1]);
		roi_weights = tf.reshape(roi_weights,[-1]);
		roi_reg_masks = tf.reshape(roi_reg_masks,[-1]);

		if self.params.box_per_class:
			rcn_regs = fully_connected(fc7_feats,4*self.num_classes,'rcn_reg', group_id=2);
			rcn_regs = tf.clip_by_value(rcn_regs,-0.2,0.2);
			rcn_bounds = [];
			for i in range(self.batch_size*roi):
				bounds.append(tf.squeeze(tf.slice(rcn_regs,[i,4*roi_classes[i]],[1, 4])));
			rcn_regs = tf.pack(bounds); 
		else:
			rcn_regs = fully_connected(fc7_feats,4,'rcn_reg',group_id=2);
			rcn_regs = tf.clip_by_value(rcn_regs,-0.2,0.2);

		loss0 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rcn_logits,labels=roi_classes)*roi_masks;
		loss0 = tf.reduce_sum(loss0*roi_weights)/tf.reduce_sum(roi_weights);
		reg_loss = l2_loss(rcn_regs,roi_regs)*roi_reg_masks;
		reg_avg = tf.reduce_sum(roi_reg_masks);
		loss0 = tf.cond(tf.less(0.0,reg_avg),lambda: loss0+self.params.rcn_relative*tf.reduce_sum(reg_loss)/reg_avg,lambda: loss0);
		loss1 = self.params.weight_decay*tf.add_n(tf.get_collection('l2_2'));
		loss = loss0+loss1;

		probabilities = tf.nn.softmax(rcn_logits);
		classes = tf.argmax(probabilities,1);
		scores = tf.reduce_max(probabilities,1); 
		scores = scores*roi_masks;

		result_classes = tf.reshape(classes,[self.batch_size,roi]);
		result_scores = tf.reshape(scores,[self.batch_size,roi]);
		if self.params.box_per_class:
			result_regs = [];
			for i in range(self.batch_size*roi):
				res_regs.append(tf.squeeze(tf.slice(rcn_regs,[i,4*classes[i]],[1, 4])));
			result_regs = tf.pack(result_regs);
		else:
			result_regs = rcn_regs;
		result_regs = tf.reshape(result_regs,[self.batch_size,roi,4]);

		self.train = train;
		self.roi_features = roi_features;
		self.roi_classes = roi_classes;
		self.roi_regs = roi_regs;
		self.roi_masks = roi_masks;
		self.roi_weights = roi_weights;
		self.roi_reg_masks = roi_reg_masks;
		self.rcn_loss = loss;
		self.rcn_loss0 = loss0;
		self.rcn_loss1 = loss1;
		self.result_classes = result_classes;
		self.result_scores = result_scores;
		self.result_regs = result_regs;
		print("RCN built......");