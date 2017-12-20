import sys;
import os;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';

import tensorflow as tf;
import argparse;
from model import *
from dataset import *
from coco.coco import *

def main(argv):
	
	parser = argparse.ArgumentParser();
	'''Model Architecture'''
	parser.add_argument('--phase', default='train', help='Train,Validate or Test');
	parser.add_argument('--load', action='store_true', default=False, help='Load the trained model');
	parser.add_argument('--cnn', default='vgg16', help='It can be VGG16 or Resnet50 or Resnet101 or Resnet152');
	parser.add_argument('--load_cnn', action='store_true', default=False, help='Load the pretrained CNN model');
	
	'''Files and Directories'''
	parser.add_argument('--cnn_file',default='./cnn/VGG16.model', help='Trained model for CNN');
	parser.add_argument('--mean_file', default='./cnn/mean.npy', help= 'Dataset mean file for Image pre-processing');
	parser.add_argument('--train_image', default='/home/arun/Projects/Tensorflow-Caption/train/images/', help='Directory containing the training images');
	parser.add_argument('--train_annotation', default='./train/instances_train2014.json', help='Captions of the training images');
	parser.add_argument('--train_anchor', default='./train/data/', help='Anchor files');
	parser.add_argument('--val_image', default='./val/images/', help='Directory containing the validation images');
	parser.add_argument('--val_annotation', default='./val/instances_val2014.json', help='Captions of the validation images')
	parser.add_argument('--val_result', default='./val/results/', help='Directory to store the validation results as images');
	parser.add_argument('--test_image', default='./test/images/', help='Directory containing the testing images');
	parser.add_argument('--result_file', default='./test/results.csv', help='File to store the testing results');
	parser.add_argument('--test_result', default='./test/results/', help='Directory to store the testing results as images');
	parser.add_argument('--save_dir', default='./models/', help='Directory to contain the trained model');
	parser.add_argument('--save_period', type=int, default=2000, help='Period to save the trained model');
	
	'''Hyper Parameters'''
	parser.add_argument('--solver', default='sgd', help='Gradient Descent Optimizer to use: Can be adam, momentum, rmsprop or sgd') 
	parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs');
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size');
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate');
	parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay');
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for some optimizers)'); 
	parser.add_argument('--decay', type=float, default=0.9, help='Decay (for some optimizers)'); 
	parser.add_argument('--batch_norm', action='store_true', default=False, help='Turn on to use batch normalization');

	'''Region Proposal parameters'''
	parser.add_argument('--roi', type=int, default=100, help='Maximum number of RoIs');
	parser.add_argument('--box_per_class', action='store_true', default=False, help='Turn on to do one bounding box regression for each class');    
	parser.add_argument('--rpn_weight', type=float, default=1.0, help='Weight for the loss of RPN');
	parser.add_argument('--rcn_weight', type=float, default=1.0, help='Weight for the loss of RCN');   
	parser.add_argument('--rpn_relative', type=float, default=10.0, help='Relative weight for bounding box regression loss vs classification loss of RPN');  
	parser.add_argument('--rcn_relative', type=float, default=10.0, help='Relative weight for bounding box regression loss vs classification loss of RCN');  
	parser.add_argument('--class_balancing_factor', type=float, default=0.8, help='Class balancing factor. The larger it is, the more attention the rare classes receive.'); 
	parser.add_argument('--prepare_anchor_data', action='store_true', default=False, help='Turn on to prepare useful anchor data for training. Must do this for the first time of training.');

	args = parser.parse_args();
	with tf.Session() as sess:
		if(args.phase=='train'):
			data = train_data(args);
			model = Model(args,'train');
			sess.run(tf.global_variables_initializer());
			if(args.load):
				model.load(sess);
			elif(args.load_cnn):
				model.load_cnn(sess,args.cnn_file);
			if(args.prepare_anchor_data):
				model.prepare_anchor_data(data);
			model.train(sess,data);
		elif(args.phase=='val'):
			data = val_data(args);
			model = Model(args,'val');
			model.load(sess);
			model.val(sess,data);
		else:
			data = test_data(args);
			model = Model(args,'test');
			model.load(sess);
			model.test(sess,data);

main(sys.argv);