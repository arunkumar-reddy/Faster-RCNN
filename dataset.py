import os;
import math;
import numpy as np;
import pandas as pd;
import pickle as pickle;

from coco.coco import *;

class_file = open('cnn/classes.pickle','r');
num_classes,classes,class_colors,class_to_category,category_to_class = pickle.load(class_file);

class Dataset():
	def __init__(self,images,files,anchors=None,categories=None,boxes=None,batch_size = 1,train = False):
		self.images = np.array(images);
		self.files = np.array(files);
		self.anchor_files = np.array(anchors);
		self.batch_size = batch_size;
		self.classes = categories;
		self.boxes = boxes;
		self.train = train;
		self.count = len(self.images);
		self.batches = int(self.count*1.0/self.batch_size);
		self.index = 0;
		self.indices = list(range(self.count));
		print('Dataset built......');
		
	def reset(self):
		self.index = 0
		np.random.shuffle(self.indices);

	def next_batch(self):
		if(self.index+self.batch_size<=self.count):
			start = self.index;
			end = self.index+self.batch_size;
			current = self.indices[start:end];
			images = self.files[current];
			if(self.train):
				anchors = self.anchors[current];
				self.index += self.batch_size;
				return images,anchors;
			else:
				self.index += self.batch_size;
				return images;

def train_data(args):
	cnn_model = args.cnn;
	image_dir = args.train_image;
	annotation_file = args.train_annotation;
	data_dir = args.train_anchor;
	batch_size = args.batch_size;
	roi = args.roi;
	coco = COCO(annotation_file);
	images = list(coco.imgToAnns.keys())
	files = [];
	anchors = [];
	boxes = [];
	categories = [];
	for image in images:
		files.append(os.path.join(image_dir,coco.imgs[image]['file_name']));
		anchors.append(os.path.join(data_dir,os.path.splitext(coco.imgs[image]['file_name'])[0]+'_'+cnn_model+'_anchor.npz')); 
		item_types = []; 
		bounds = []; 
		for item in coco.imgToAnns[image]: 
			item_types.append(category_to_class[item['category_id']]); 
			bounds.append([item['bbox'][1],item['bbox'][0],item['bbox'][3]+1,item['bbox'][2]+1]);
		categories.append(item_types);
		boxes.append(bounds);
	dataset = Dataset(images,files,anchors,categories,boxes,batch_size,True);
	return dataset;

def val_data(args):
	cnn_model = args.cnn;
	image_dir = args.val_image;
	annotation_file = args.val_annotation;
	coco = COCO(annotation_file);
	images = list(coco.imgToAnns.keys());
	files = [];
	for image in images:
		files.append(os.path.join(image_dir,coco.imgs[image]['file_name']));
	dataset = Dataset(images,files);
	return dataset;

def test_data(args):
	image_dir = args.test_image;
	files = os.listdir(image_dir);
	images = list(range(len(files)));
	dataset = Dataset(images,files);
	return dataset;