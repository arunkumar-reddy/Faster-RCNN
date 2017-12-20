import tensorflow as tf;
import numpy as np;
import cv2;
import os;
import matplotlib.pyplot as plt;
import matplotlib.image as mpimg;
from tqdm import tqdm;
from dataset import *;
from cnn import *;
from rpn import *;
from rcn import *;
from bbox import *;
from coco.coco import *;
from coco.cocoeval import *;

class Loader(object):
	def __init__(self,mean_file):
		self.rgb = True;
		self.scale_shape = np.array([640,640],np.int32);
		self.crop_shape = np.array([640,640],np.int32);
		self.mean = np.load(mean_file).mean(1).mean(1);

	def load(self,files):
		images = [];
		for image_file in files:
			image_file = '/home/arun/Projects/Tensorflow-Caption/'+image_file;
			image = cv2.imread(image_file);
			if(self.rgb):
				temp = image.swapaxes(0,2);
				temp = temp[::-1];
				image = temp.swapaxes(0,2);
			image = cv2.resize(image,(self.scale_shape[0],self.scale_shape[1]));
			offset = (self.scale_shape-self.crop_shape)/2;
			offset = offset.astype(np.int32);
			image = image[offset[0]:offset[0]+self.crop_shape[0], offset[1]:offset[1]+self.crop_shape[1],:];
			image = image-self.mean;
			images.append(image);
		images = np.array(images,np.float32);
		return images;

class Model(object):
	def __init__(self,params,phase):
		self.params = params;
		self.phase = phase;
		self.cnn_model = params.cnn;
		self.roi = params.roi;
		self.box_per_class = params.box_per_class;
		self.num_classes = num_classes;
		self.classes = classes;
		self.class_colors = class_colors;
		self.class_to_category = class_to_category;
		self.category_to_class = category_to_class;
		self.background_id = self.num_classes-1;
		self.batch_size = params.batch_size if phase=='train' else 1;
		self.save_dir = os.path.join(params.save_dir,self.cnn_model+'/');
		self.imageloader = Loader(params.mean_file);
		self.image_shape = [640,640,3];
		self.anchor_scales = [50,100,200,300,400,500]; 
		self.anchor_ratios = [[1.0/math.sqrt(2),math.sqrt(2)], [1.0,1.0], [math.sqrt(2),1.0/math.sqrt(2)]];
		self.num_anchor_type = len(self.anchor_scales)*len(self.anchor_ratios);
		self.anchor_shapes = [];
		for scale in self.anchor_scales:
			for ratio in self.anchor_ratios:
				self.anchor_shapes.append([int(scale*ratio[0]), int(scale*ratio[1])]);
		self.anchor_stat_file = self.cnn_model+'_anchor_stats.npz';
		self.global_step = tf.Variable(0,name='global_step',trainable = False); 
		self.saver = tf.train.Saver(max_to_keep = 100);
		self.build();

	def build(self):
		print('Building the Model......');
		cnn = CNN(self.params);
		self.feature_shape = cnn.feature_shape;
		if(self.cnn_model=='vgg16'):
			self.roi_feature_shape = [16,16,512];
			self.roi_pooled_shape = [8,8,512];
		else:
			self.roi_feature_shape = [10,10,2048];
			self.roi_pooled_shape = [5,5,2048];
		self.build_anchors();
		rpn = RPN(self.params,self.total_num_anchor);
		rcn = RCN(self.params,self.roi_feature_shape,num_classes);
		self.cnn = cnn;
		self.rpn = rpn;
		self.rcn = rcn;

		loss0 = self.params.rpn_weight*rpn.rpn_loss0 + self.params.rcn_weight*rcn.rcn_loss0;
		loss1 = self.params.weight_decay*(tf.add_n(tf.get_collection('l2_1'))+tf.add_n(tf.get_collection('l2_2')));
		loss = loss0+loss1;
		if self.params.solver == 'adam':
			solver = tf.train.AdamOptimizer(self.params.learning_rate);
		elif self.params.solver == 'momentum':
			solver = tf.train.MomentumOptimizer(self.params.learning_rate,self.params.momentum);
		elif self.params.solver == 'rmsprop':
			solver = tf.train.RMSPropOptimizer(self.params.learning_rate,self.params.decay,self.params.momentum);
		else:
			solver = tf.train.GradientDescentOptimizer(self.params.learning_rate);

		optimizer = solver.minimize(loss, global_step=self.global_step);
		self.loss = loss;
		self.loss0 = loss0;
		self.loss1 = loss1;
		self.optimizer = optimizer;
		print('Model completed......');

	def train(self,sess,data):
		print('Training the model.......');
		self.setup();
		epochs = self.params.epochs;
		for epoch in tqdm(list(range(epochs)),desc='Epoch'):
			for i in tqdm(list(range(data.batches)),desc='Batch'):
				batch = data.next_batch();
				files,_ = batch;
				images = self.imageloader.load(files);
				features = sess.run(self.cnn.features,feed_dict={self.cnn.images:images,self.cnn.train:True });
				batch_data = self.feed(batch,train=True,features = features);
				_, loss0, loss1, global_step = sess.run([self.optimizer,self.loss0,self.loss1,self.global_step],feed_dict=batch_data);
				print(" Loss0=%f Loss1=%f" %(loss0,loss1));
				if ((global_step+1)%self.params.save_period)==0:
					self.save(sess);
			data.reset();
		self.save(sess);
		print('Training completed......');

	def val(self,sess,data):
		print('Validating the Model......');
		num_roi = self.roi;
		detected_scores = [];
		detected_classes = [];
		detected_boxes = [];
		for i in tqdm(list(range(data.count)),desc='Batch'):
			batch = data.next_batch();
			files = batch;
			images = self.imageloader.load(files);
			features = sess.run(self.cnn.features,feed_dict={self.cnn.images:images,self.cnn.train:False });
			scores,regs = sess.run([self.rpn.rpn_scores,self.rpn.rpn_regs], feed_dict={self.rpn.cnn_features:features,self.rpn.train:False});
			rois = unparam_bbox(regs.squeeze(),self.anchors,self.image_shape[:2]);
			num_real_roi,real_rois = self.process_rpn_result(scores.squeeze(),rois);
			rois = np.ones((num_roi,4),np.int32)*3;
			rois[:num_real_roi] = real_rois;
			rois = expand_bbox(rois,self.image_shape[:2])
			rois = np.expand_dims(expanded_rois,0);
			masks = np.zeros((num_roi),np.float32);
			masks[:num_real_roi] = 1.0;
			masks = np.expand_dims(masks,0);
			batch_data = process_roi_data(train=False,features=features,rois=rois,masks=masks);
			scores,categories,regs = sess.run([self.rcn.result_scores,self.rcn.result_classes,self.rcn.result_regs],feed_dict=batch_data);
			boxes = unparam_bbox(regs.squeeze(),rois);
			detected_count,scores,classes,boxes = self.process_rcn_result(scores.squeeze(),classes.squeeze(),boxes);
			detected_scores.append(scores);
			detected_classes.append(classes);
			detected_boxes.append(boxes);
		data.reset();
		results = [];
		for i in range(data.count): 
			for s, c, b in zip(detected_scores[i],detected_classes[i],detected_boxes[i]): 
				results.append({'image_id': data.images[i], 'category_id': self.class_to_category[c], 'bbox':[b[1],b[0],b[3]-1,b[2]-1], 'score': s}); 

		val_res_coco = val_coco.loadRes2(results);
		scorer = COCOeval(val_coco,val_res_coco); 
		scorer.evaluate();
		scorer.accumulate();
		scorer.summarize(); 
		print('Validation complete......');

	def test(self,sess,data):
		pass;

	def build_anchors(self):
		'''Build the anchors and their parents which include the surrounding contexts'''
		image_shape = np.array(self.image_shape[:2],np.int32);
		feature_shape = np.array(self.feature_shape[:2],np.int32); 
		for i in range(3):
			for j in range(3):
				num_anchor, anchors, anchor_is_untruncated, num_untruncated_anchor, parent_anchors, parent_anchor_is_untruncated, num_untruncated_parent_anchor = generate_anchors(image_shape, feature_shape, self.anchor_scales[i], self.anchor_ratios[j]);
				if i==0 and j==0:
					self.num_anchor = num_anchor;
					self.anchors = anchors;
					self.anchor_is_untruncated = anchor_is_untruncated;
					self.num_untruncated_anchor = num_untruncated_anchor;
					self.parent_anchors = parent_anchors;
					self.parent_anchor_is_untruncated = parent_anchor_is_untruncated;
					self.num_untruncated_parent_anchor = num_untruncated_parent_anchor;
				else:
					self.num_anchor = np.concatenate((self.num_anchor, num_anchor));
					self.anchors = np.concatenate((self.anchors, anchors));
					self.anchor_is_untruncated = np.concatenate((self.anchor_is_untruncated, anchor_is_untruncated));
					self.num_untruncated_anchor = np.concatenate((self.num_untruncated_anchor, num_untruncated_anchor));
					self.parent_anchors = np.concatenate((self.parent_anchors, parent_anchors));
					self.parent_anchor_is_untruncated = np.concatenate((self.parent_anchor_is_untruncated, parent_anchor_is_untruncated));
					self.num_untruncated_parent_anchor = np.concatenate((self.num_untruncated_parent_anchor, num_untruncated_parent_anchor));

		feature_shape = (feature_shape/2).astype(np.int32) 
		for i in range(3, 6):
			for j in range(3):
				num_anchor, anchors, anchor_is_untruncated, num_untruncated_anchor, parent_anchors, parent_anchor_is_untruncated, num_untruncated_parent_anchor = generate_anchors(image_shape, feature_shape, self.anchor_scales[i], self.anchor_ratios[j]);
				self.num_anchor = np.concatenate((self.num_anchor, num_anchor));
				self.anchors = np.concatenate((self.anchors, anchors));
				self.anchor_is_untruncated = np.concatenate((self.anchor_is_untruncated, anchor_is_untruncated));
				self.num_untruncated_anchor = np.concatenate((self.num_untruncated_anchor, num_untruncated_anchor));
				self.parent_anchors = np.concatenate((self.parent_anchors, parent_anchors));
				self.parent_anchor_is_untruncated = np.concatenate((self.parent_anchor_is_untruncated, parent_anchor_is_untruncated));
				self.num_untruncated_parent_anchor = np.concatenate((self.num_untruncated_parent_anchor, num_untruncated_parent_anchor));
 
		self.total_num_anchor = np.sum(self.num_anchor);
		self.total_num_untruncated_anchor = np.sum(self.num_untruncated_anchor);
		self.total_num_truncated_anchor = self.total_num_anchor - self.total_num_untruncated_anchor;
		'''
		for i in range(self.num_anchor_type):
			print("Anchor type [%d, %d]: %d untruncated, %d truncated" %(self.anchor_shapes[i][0], self.anchor_shapes[i][1], self.num_untruncated_anchor[i], self.num_anchor[i]-self.num_untruncated_anchor[i]));
		'''
		print("Anchors built......");

	def load(self,sess):
		print('Loading model......');
		checkpoint = tf.train.get_checkpoint_state(self.save_dir);
		if checkpoint is None:
			print('Error: No saved model found. Please train first......');
			sys.exit(0);
		self.saver.restore(sess, checkpoint.model_checkpoint_path);

	def load_cnn(self,sess,cnn_path):
		print('Loading CNN model from %s' %data_path);
		data_dict = np.load(cnn_path).item();
		count = 0;
		miss_count = 0;
		for op_name in data_dict:
			with tf.variable_scope(op_name, reuse=True):
				for param_name, data in data_dict[op_name].iteritems():
					try:
						var = tf.get_variable(param_name);
						session.run(var.assign(data));
						count += 1;
					except ValueError:
						miss_count += 1;
						if not ignore_missing:
							raise
		print('%d variables loaded. %d variables missed......' %(count, miss_count));

	def save(self, sess):
		print(('Saving model to %s' % self.save_dir));
		self.saver.save(sess,self.save_dir,self.global_step);

	def feed(self,batch,train,features=None):
		if(train):
			_, anchor_files = batch;
			anchor_labels,anchor_regs,anchor_masks,anchor_weights,anchor_reg_masks = self.process_anchor_data(anchor_files);
			rois,roi_classes,roi_regs,roi_masks,roi_weights,roi_reg_masks = self.process_roi_data(anchor_files);
			rois = rois.reshape((-1,4));
			rois = convert_bbox(rois,self.image_shape[:2],self.feature_shape[:2]);
			rois = rois.reshape((self.batch_size,self.roi,4));
			roi_features = self.get_roi_feats(features,rois);
			return {self.rpn.cnn_features: features, self.rpn.anchor_labels: anchor_labels, self.rpn.anchor_regs: anchor_regs, self.rpn.anchor_masks: anchor_masks, self.rpn.anchor_weights: anchor_weights, self.rpn.anchor_reg_masks: anchor_reg_masks, self.rcn.roi_features: roi_features, self.rcn.roi_classes: roi_classes, self.rcn.roi_regs: roi_regs, self.rcn.roi_masks: roi_masks, self.rcn.roi_weights: roi_weights, self.rcn.roi_reg_masks: roi_reg_masks, self.rpn.train: train, self.rcn.train: train};
		else:
			files = batch;
			images = self.imageloader.load(files);
			return {self.cnn.images: images,self.cnn.train: train};	

	def process_roi_data(self,train,features,rois,masks):
		rois = rois.reshape((-1,4));
		rois = convert_bbox(rois,self.image_shape[:2],self.feature_shape[:2]);
		rois = rois.reshape((self.batch_size,self.roi,4));
		roi_features = self.get_roi_feats(features,rois);
		return {self.rcn.roi_features: roi_features, self.rcn.roi_masks: masks, self.rcn.train: train};

	def setup(self):
		p = self.params.class_balancing_factor;
		stats = np.load(self.anchor_stat_file);
		self.anchor_iou_freq = stats['anchor_iou_freq'];
		self.class_iou_freq = stats['class_iou_freq'];
		self.anchor_iou_weight = np.exp(-np.log(self.anchor_iou_freq)*p); 
		self.anchor_iou_weight[np.where(self.anchor_iou_weight>1e5)] = 0;
		self.anchor_iou_weight[:, :3, :] *= 0.2;
		M = np.sum(self.class_iou_freq[:-1,4:,:])*1.0;
		K = np.sum(self.class_iou_freq[-1,:3,:])*1.0;
		self.num_object = min(M,self.num_roi*0.6);
		self.num_background = min(K,self.num_roi*0.4);
		self.obj_filter_rate = self.num_object/M;
		self.bg_filter_rate = self.num_background/K;
		self.class_iou_weight = np.exp(-np.log(self.class_iou_freq*self.obj_filter_rate)*p); 
		self.class_iou_weight[-1] = np.exp(-np.log(self.class_iou_freq[-1]*self.bg_filter_rate)*p)*0.2;
		self.class_iou_weight[np.where(self.class_iou_weight>1e5)] = 0;
			 
	def process_anchor_data(self, anchor_files): 
		anchor_labels = [];
		anchor_regs = [];
		anchor_masks = [];
		anchor_weights = [];
		anchor_reg_masks = [];
		types = self.num_anchor_type;
		for i in range(self.batch_size):
			anchor_data = np.load(anchor_files[i]);
			labels = anchor_data['labels'];
			regs = anchor_data['regs'];
			ious = anchor_data['ious'];
			ioas = anchor_data['ioas'];
			iogs = anchor_data['iogs'];
			start = 0;
			masks = np.array([]);
			weights = np.array([]);
			reg_masks = np.array([]);
			for j in range(types):
				end = start + self.num_anchor[j];
				current_labels = labels[start:end];
				current_ious = ious[start:end];
				current_ioas = ioas[start:end];
				current_iogs = iogs[start:end];
				flags = self.anchor_is_untruncated[start:end];

				idx1 = np.array(np.floor((current_ious-0.01)/0.2)+1,np.int32); 
				max_ioa_iogs = np.maximum(current_ioas,current_iogs);
				idx2 = np.array(np.floor((max_ioa_iogs-0.01)/0.2)+1,np.int32);

				current_masks = np.zeros((self.num_anchor[j]),np.float32);
				current_weights = np.zeros((self.num_anchor[j]),np.float32);
				current_reg_masks = np.zeros((self.num_anchor[j]),np.float32);

				for k in range(self.num_anchor[j]):
					current_masks[k] = flags[k];
					current_weights[k] = flags[k]*self.anchor_iou_weight[j,idx1[k],idx2[k]];
					current_reg_masks[k] = flags[k]*self.anchor_iou_weight[j,idx1[k],idx2[k]]*(current_labels[k]==1);

				masks = np.concatenate((masks,current_masks));
				weights = np.concatenate((weights,current_weights));
				reg_masks = np.concatenate((reg_masks,current_reg_masks));
				start = end;

			labels[np.where(labels==-1)[0]] = 0;
			anchor_labels.append(labels);
			anchor_regs.append(regs);
			anchor_masks.append(masks);
			anchor_weights.append(weights);
			anchor_reg_masks.append(reg_masks);

		gt_anchor_labels = np.array(gt_anchor_labels);
		gt_anchor_regs = np.array(gt_anchor_regs);
		anchor_masks = np.array(anchor_masks);
		anchor_weights = np.array(anchor_weights);
		anchor_reg_masks = np.array(anchor_reg_masks);
		return anchor_labels, anchor_regs, anchor_masks, anchor_weights, anchor_reg_masks; 

	def get_roi_features(self,features,rois):
		roi_features = [];
		for i in range(self.batch_size):
			roi_features.append(self.roi_warp(features[i],rois[i]));
		roi_features = np.array(roi_features);
		return roi_features;

	def roi_warp(self,feats,rois):
		ch,cw,c = self.feature_shape;
		th,tw,c = self.roi_feature_shape;
		num_roi = self.roi;
		warped_features = [];
		for k in range(num_roi):
			y, x, h, w = rois[k,0], rois[k,1], rois[k,2], rois[k,3];
			j = np.array(list(range(h)),np.float32);
			i = np.array(list(range(w)),np.float32);
			tj = np.array(list(range(th)),np.float32);
			ti = np.array(list(range(tw)),np.float32);
			j = np.expand_dims(np.expand_dims(np.expand_dims(j,1),2),3);
			i = np.expand_dims(np.expand_dims(np.expand_dims(i,0),2),3);
			tj = np.expand_dims(np.expand_dims(np.expand_dims(tj,1),0),1);
			ti = np.expand_dims(np.expand_dims(np.expand_dims(ti,0),0),1);
			j = np.tile(j,(1,w,th,tw));
			i = np.tile(i,(h,1,th,tw)); 
			tj = np.tile(tj,(h,w,1,tw)) 
			ti = np.tile(ti,(h,w,th,1));
			b = tj*h*1.0/th-j
			a = ti*w*1.0/tw-i;
			b = np.maximum(np.zeros_like(b),1-np.absolute(b));
			a = np.maximum(np.zeros_like(a),1-np.absolute(a));
			G = b*a;
			G = G.reshape((h*w,th*tw));
			sliced_feature = features[y:y+h,x:x+w,:];
			sliced_feature = sliced_feature.swapaxes(0,1);
			sliced_feature = sliced_feature.swapaxes(0,2);
			sliced_feature = sliced_feature.reshape((-1, h*w));

			warped_feature = np.matmul(sliced_features,G);
			warped_feature = warped_feature.reshape((-1,th,tw));
			warped_feature = warped_feature.swapaxes(0,1);
			warped_feature = warped_feature.swapaxes(1,2);
			warped_feats.append(warped_feature);

		warped_features = np.array(warped_features);
		return warped_features;

	def process_roi_data(self, anchor_files):
		num_roi = self.roi;
		rois = [];
		roi_classes = [];
		roi_regs = [];
		roi_masks = [];
		roi_weights = [];
		roi_reg_masks = [];
		X = self.num_object;
		Y = self.num_background;

		for i in range(self.batch_size):
			anchor_data = np.load(anchor_files[i]);
			labels = anchor_data['labels'];
			regs = anchor_data['regs'] ;
			classes = anchor_data['classes'];
			ious = anchor_data['ious'];
			ioas = anchor_data['ioas'];
			iogs = anchor_data['iogs'];
			sorted_idx = anchor_data['sorted_idx']
			
			A = len(np.where(labels==1)[0]);
			B = len(np.where(labels==0)[0]);
			C = self.total_num_truncated_anchor;
			U = min(X,A);
			V = min(Y,B);          

			if U>0:
				p = int(A*1.0/U);
				f = int(np.random.uniform(0, 1)*p);
				obj_idx = np.array(list(range(f,A,p)),np.int32);
			else:
				obj_idx = np.array([],np.int32);      

			if V>0:
				q = int(B*1.0/V);
				g = int(np.random.uniform(0, 1)*q);  
				bg_idx = -np.array(list(range(g+C+1,B+C+1,q)),np.int32);
			else:
				bg_idx = np.array([],np.int32);

			chosen_idx = np.concatenate((obj_idx,bg_idx));
			chosen_idx = sorted_idx[chosen_idx];
			num_real_roi = len(chosen_idx);
			real_rois = self.parent_anchors[chosen_idx];
			real_roi_regs = regs[chosen_idx];
			real_roi_classes = classes[chosen_idx];
			real_roi_ious = ious[chosen_idx];
			real_roi_ioas = ioas[chosen_idx];
			real_roi_iogs = iogs[chosen_idx];

			idx1 = np.array(np.floor((real_roi_ious-0.01)/0.2)+1,np.int32); 
			max_ioa_iogs = np.maximum(real_roi_ioas,real_roi_iogs);
			idx2 = np.array(np.floor((max_ioa_iogs-0.01)/0.2)+1,np.int32); 

			real_roi_masks = np.ones((num_real_roi),np.float32);
			real_roi_weights = np.zeros((num_real_roi),np.float32);
			real_roi_reg_masks = np.zeros((num_real_roi),np.float32);

			for k in range(num_real_roi):
				real_roi_weights[k] = self.class_iou_weight[real_roi_classes[k],idx1[k],idx2[k]];
				real_roi_reg_masks[k] = self.class_iou_weight[real_roi_classes[k],idx1[k],idx2[k]]*(real_roi_classes[k]!=self.background_id); 

			current_rois = np.ones((num_roi,4),np.int32)*3;
			current_rois[:num_real_roi] = real_rois;
			current_roi_classes = np.ones((num_roi),np.int32); 
			current_roi_classes[:num_real_roi] = real_roi_classes; 
			current_roi_regs = np.ones((num_roi,4),np.float32);
			current_roi_regs[:num_real_roi] = real_roi_regs;
			current_roi_masks = np.zeros((num_roi),np.float32); 
			current_roi_masks[:num_real_roi] = real_roi_masks;
			current_roi_weights = np.zeros((num_roi),np.float32);
			current_roi_weights[:num_real_roi] = real_roi_weights;
			current_roi_reg_masks = np.zeros((num_roi),np.float32);
			current_roi_reg_masks[:num_real_roi] = real_roi_reg_masks; 

			rois.append(current_rois);
			roi_classes.append(current_roi_classes);
			roi_regs.append(current_roi_regs);
			roi_masks.append(current_roi_masks);
			roi_weights.append(current_roi_weights); 
			roi_reg_masks.append(current_roi_reg_masks); 

		rois = np.array(rois);
		roi_classes = np.array(gt_roi_classes);
		roi_regs = np.array(gt_roi_regs);
		roi_masks = np.array(roi_masks);
		roi_weights = np.array(roi_weights);
		roi_reg_masks = np.array(roi_reg_masks);
		return rois, roi_classes, roi_regs, roi_masks, roi_weights, roi_reg_masks;

	def prepare_anchor_data(self,dataset):
		print('Labeling the anchors......');
		num_types = self.num_anchor_type;
		num_classes = self.num_classes;
		anchor_iou_freq = np.zeros((num_types,6,6),np.float32);
		class_iou_freq = np.zeros((num_classes,6,6),np.float32);
		for i in tqdm(list(range(dataset.count))):
			image = dataset.files[i];
			classes = np.array(dataset.classes[i]) 
			boxes = np.array(dataset.boxes[i])
			boxes = convert_bbox(boxes,[640,480],self.image_shape[:2]);
			labels,boxes,classes,ious,ioas,iogs = label_anchors(self.anchors,self.anchor_is_untruncated,classes,boxes,self.background_id);
			start = 0;
			for j in range(num_types):
				end = start + self.num_anchor[j];
				current_labels = labels[start:end];
				current_classes = classes[start:end];
				current_ious = ious[start:end];
				current_ioas = ioas[start:end];
				current_iogs = iogs[start:end];
				flags = self.anchor_is_untruncated[start:end];
				idx1 = np.array(np.floor((current_ious-0.01)/0.2)+1,np.int32); 
				max_ioa_iogs = np.maximum(current_ioas,current_iogs)
				idx2 = np.array(np.floor((max_ioa_iogs-0.01)/0.2)+1,np.int32); 
				for k in range(self.num_anchor[j]):                    
					anchor_iou_freq[j,idx1[k],idx2[k]] += flags[k];
					class_iou_freq[current_classes[k],idx1[k],idx2[k]] += flags[k];
				start = end;

			sorted_idx = np.argsort(ious)[::-1];
			num_hit = len(np.where(labels==1)[0]);  
			regs = param_bbox(boxes,self.anchors);
			np.savez(dataset.anchor_files[i],labels=labels,boxes=boxes,regs=regs,classes=classes,ious=ious,ioas=ioas,iogs=iogs,sorted_idx=sorted_idx);

		self.anchor_iou_freq = (anchor_iou_freq+0.001)/dataset.count;
		self.class_iou_freq = (class_iou_freq+0.001)/dataset.count;
		np.savez(self.anchor_stat_file,anchor_iou_freq=self.anchor_iou_freq,class_iou_freq=self.class_iou_freq);