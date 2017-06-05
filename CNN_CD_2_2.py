import tensorflow as tf
import numpy as np
import os
import time
from time import gmtime, strftime

'2009-01-05 22:14:39'
#Flavia Rawsize 1600 * 1200 
#IMG_W = 160  # resize the image, if the input image is too large, training will be very slow.
#IMG_H = 120 #

RATIO = 0.2 # take 20% of dataset as validation data 

BATCH_SIZE = 32 #64
CAPACITY = 1907 # total 1907 images
MAX_STEP = 6000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001

N_CLASSES = 32
# set data directory
train_dir = '/home/shi/Documents/PR970/Leaves/'

nameDict = {'pb':"pubescentbamboo", 
			'chc':"Chinese horse chestnut",
			'ab':"Anhui Barberry",
			'cr':"Chinese redbud",
			'ti':"true indigo",
			'jm':"Japanese maple",
			'nan':"Nanmu",
			'ca':"castor aralia",
			'cc':"Chinese cinnamon",
			'gt':"goldenrain tree",

			'bfh':"Big-fruited Holly",
			'jc':  "Japanese cheesewood",
			'ws':"wintersweet",
			'cam': "camphortree",
			'ja':"Japan Arrowwood",
			'so':"sweet osmanthus",
			'deo':"deodar",
			'mt':"ginkgo, maidenhair tree",
			'cm':"Crape myrtle, Crepe myrtle",
			'ole':  "oleander",

			'ypp':"yew plum pine",
			'jfc':"Japanese Flowering Cherry",
			'gp':"Glossy Privet",
			'ct':"Chinese Toon",
			'pea':"peach",
			'fw':"Ford Woodlotus",
			'tm':"trident maple",
			'bb':"Beale's barberry",
			'sm':"southern magnolia",
			'cp':"Canadian poplar",
			'ctt':"Chinese tulip tree",
			'tan':"tangerine"}
# nameDict2 = {
#             'l1' :"Ulmus carpinifolia"
#             #...

# }   

# def decoder2(code):
#     code = code[0:-9]
#     if code == 'l1':
#         leaf = "l1"
	




def decoder(code):
	"""input 4 digits flavia image name
		output a leaf abbreviation str."""
	code = int(code)
	if code >= 1001 and code <= 1059:
		leaf = 'pb'
	elif code >= 1060 and code <= 1122:
		leaf = 'chc'           
	elif code >= 1552 and code <= 1616:
		leaf = 'ab'
	elif code >= 1123 and code <= 1194:
		leaf = 'cr'
	elif code >= 1195 and code <= 1267:
		leaf = 'ti'
	elif code >= 1268 and code <= 1323:
		leaf = 'jm'
	elif code >= 1324 and code <= 1385:
		leaf = 'nan'
	elif code >= 1386 and code <= 1437:
		leaf = 'ca'
	elif code >= 1497 and code <= 1551:
		leaf = 'cc'
	elif code >= 1438 and code <= 1496:
		leaf = 'gt'
	elif code >= 2001 and code <= 2050:
		leaf = 'bfh'
	elif code >= 2051 and code <= 2113:
		leaf = 'jc'
	elif code >= 2114 and code <= 2165:
		leaf = 'ws'
	elif code >= 2166 and code <= 2230:
		leaf = 'cam'
	elif code >= 2231 and code <= 2290:
		leaf = 'ja'
	elif code >= 2291 and code <= 2346:
		leaf = 'so'
	elif code >= 2347 and code <= 2423:
		leaf = 'deo'
	elif code >= 2424 and code <= 2485:
		leaf = 'mt'
	elif code >= 2486 and code <= 2546:
		leaf = 'cm'
	elif code >= 2547 and code <= 2612:
		leaf = 'ole'
	elif code >= 2616 and code <= 2675:
		leaf = 'ypp'
	elif code >= 3001 and code <= 3055:
		leaf = 'jfc'
	elif code >= 3056 and code <= 3110:
		leaf = 'gp'    
	elif code >= 3111 and code <= 3175:
		leaf = 'ct'
	elif code >= 3176 and code <= 3229:
		leaf = 'pea'
	elif code >= 3230 and code <= 3281:
		leaf = 'fw'
	elif code >= 3282 and code <= 3334:
		leaf = 'tm'
	elif code >= 3335 and code <= 3389:
		leaf = 'bb'
	elif code >= 3390 and code <= 3446:
		leaf = 'sm'
	elif code >= 3447 and code <= 3510:
		leaf = 'cp'
	elif code >= 3511 and code <= 3563:
		leaf = 'ctt'
	elif code >= 3566 and code <= 3621:
		leaf = 'tan'
	else:
		print "Check the input code, it is not in the database"
		return
	return leaf

def get_files(file_dir, ratio):
	'''
	Args:
		file_dir: file directory
	Returns:
		list of images and labels
	'''
	variDict = {}
	labDict = {}

	#how many total classes
	classDict = {}
	for k, v in enumerate(nameDict):
		variDict[v] = []
		labDict[v]  = []
		classDict[v] = k
	#print variDict
	#print labDict
	print classDict
	#return 
	for file in os.listdir(file_dir):
		code = int(file[0:4])
		leaf = decoder(code)
		variDict[leaf].append(file_dir+file)
		labDict[leaf].append(classDict[leaf])

	#stack all leaves together
	#initilize image_list with the first one pb
	image_list = variDict['pb']
	label_list = labDict['pb']
	for key in variDict:
		if key == 'pb':
			continue
		else:
			image_list = np.hstack((image_list, variDict[key]))
			label_list = np.hstack((label_list, labDict[key]))
	#print image_list.shape
	#print label_list.shape

	temp = np.array([image_list, label_list])
	temp = temp.transpose()

	#ramdomization
	np.random.seed(1)
	np.random.shuffle(temp)   
	
	all_image_list = temp[:, 0]
	all_label_list = temp[:, 1]
	
	#test the load
	#print all_image_list[0]
	#print all_label_list[0]
	#testcode = int(all_image_list[0][-8:-4])
	#testleaf = decoder(testcode)
	#print nameDict[testleaf], labDict[testleaf]
	#return

	n_sample = len(all_label_list) 
	n_train = int(n_sample - n_sample*ratio) # number of trainning samples
	tra_images = all_image_list[0:n_train]
	tra_labels = all_label_list[0:n_train]
	tra_labels = [int(float(i)) for i in tra_labels]
	val_images = all_image_list[n_train:-1]
	val_labels = all_label_list[n_train:-1]
	val_labels = [int(float(i)) for i in val_labels]
	
	return tra_images,tra_labels,val_images,val_labels

#get_files(train_dir, 0.2)

def get_batch(image, label, image_W, image_H, batch_size, capacity):
	'''
	Args:
		image: list type
		label: list type
		image_W: image width
		image_H: image height
		batch_size: batch size
		capacity: the maximum elements in queue
	Returns:
		image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
		label_batch: 1D tensor [batch_size], dtype=tf.int32
	'''
	image = tf.cast(image, tf.string)
	label = tf.cast(label, tf.int32)

	# make an input queue
	input_queue = tf.train.slice_input_producer([image, label])
	
	label = input_queue[1]
	image_contents = tf.read_file(input_queue[0])
	image = tf.image.decode_jpeg(image_contents, channels = 3) # channels: 1 :grayscale, 3 :RGB
	
	######################################
	# data argumentation should go to here
	######################################
	
	#image = tf.image.resize_image(image, image_W, image_H)


	####resize#############################################################
	#######################    image processing  ##########################
	#######################################################################
	image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)   


	#######################################################################
	# if you want to test the generated batches of images, you might want to comment the following line.
	
	image = tf.image.per_image_standardization(image)
	
	image_batch, label_batch = tf.train.batch([image, label],
												batch_size= batch_size,
												num_threads= 64, 
												capacity = capacity)
	
	label_batch = tf.reshape(label_batch, [batch_size])
	image_batch = tf.cast(image_batch, tf.float32)
	
	return image_batch, label_batch


def inference(images, batch_size, n_classes):
	'''Build the model
	Args:
		images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
	Returns:
		output tensor with the computed logits, float, [batch_size, n_classes]
	'''
	#conv1, shape = [kernel size, kernel size, channels, kernel numbers]
	
	con1_kernel = 8 #16

	with tf.variable_scope('conv1') as scope:
		weights = tf.get_variable('weights', 
								  shape = [3,3,3,con1_kernel],# sizex, sizey, RGB channel, nodes
								  dtype = tf.float32, 
								  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
		biases = tf.get_variable('biases', 
								 shape=[con1_kernel],
								 dtype=tf.float32,
								 initializer=tf.constant_initializer(0.1))
		conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name= scope.name)
	
	#pool1 and norm1   
	with tf.variable_scope('pooling1_lrn') as scope:
		pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
							   padding='SAME', name='pooling1')
		norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
						  beta=0.75,name='norm1')
	
	#conv2

	con2_kernel = 8 #16

	with tf.variable_scope('conv2') as scope:
		weights = tf.get_variable('weights',
								  shape=[3,3,con1_kernel,con2_kernel],
								  dtype=tf.float32,
								  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
		biases = tf.get_variable('biases',
								 shape=[con2_kernel], 
								 dtype=tf.float32,
								 initializer=tf.constant_initializer(0.1))
		conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name='conv2')
	
	
	#pool2 and norm2
	with tf.variable_scope('pooling2_lrn') as scope:
		norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
						  beta=0.75,name='norm2')
		pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],
							   padding='SAME',name='pooling2')
	
	
	#local3

	l3_fully_connected_node = 64 #128

	with tf.variable_scope('local3') as scope:
		reshape = tf.reshape(pool2, shape=[batch_size, -1])
		dim = reshape.get_shape()[1].value
		weights = tf.get_variable('weights',
								  shape=[dim,l3_fully_connected_node],
								  dtype=tf.float32,
								  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
		biases = tf.get_variable('biases',
								 shape=[l3_fully_connected_node],
								 dtype=tf.float32, 
								 initializer=tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)    
	
	#local4

	l4_fully_connected_node = 64 #128

	with tf.variable_scope('local4') as scope:
		weights = tf.get_variable('weights',
								  shape=[l3_fully_connected_node,l4_fully_connected_node],
								  dtype=tf.float32, 
								  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
		biases = tf.get_variable('biases',
								 shape=[l4_fully_connected_node],
								 dtype=tf.float32,
								 initializer=tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
	 
		
	# softmax
	with tf.variable_scope('softmax_linear') as scope:
		weights = tf.get_variable('softmax_linear',
								  shape=[l4_fully_connected_node, n_classes],
								  dtype=tf.float32,
								  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
		biases = tf.get_variable('biases', 
								 shape=[n_classes],
								 dtype=tf.float32, 
								 initializer=tf.constant_initializer(0.1))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
	
	return softmax_linear

def losses(logits, labels):
	'''Compute loss from logits and labels
	Args:
		logits: logits tensor, float, [batch_size, n_classes]
		labels: label tensor, tf.int32, [batch_size]
		
	Returns:
		loss tensor of float type
	'''
	with tf.variable_scope('loss') as scope:
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
						(logits=logits, labels=labels, name='xentropy_per_example')
		loss = tf.reduce_mean(cross_entropy, name='loss')
		tf.summary.scalar(scope.name+'/loss', loss)
	return loss

def trainning(loss, learning_rate):
	'''Training ops, the Op returned by this function is what must be passed to 
		'sess.run()' call to cause the model to train.
		
	Args:
		loss: loss tensor, from losses()
		
	Returns:
		train_op: The op for trainning
	'''
	with tf.name_scope('optimizer'):
		optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step= global_step)
	return train_op

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
	logits: Logits tensor, float - [batch_size, NUM_CLASSES].
	labels: Labels tensor, int32 - [batch_size], with values in the
	  range [0, NUM_CLASSES).
  Returns:
	A scalar int32 tensor with the number of examples (out of batch_size)
	that were predicted correctly.
  """
  with tf.variable_scope('accuracy') as scope:
	  correct = tf.nn.in_top_k(logits, labels, 1)
	  correct = tf.cast(correct, tf.float16)
	  accuracy = tf.reduce_mean(correct)
	  tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy

def run_training(IMG_W,IMG_H):

	#IMG_W = 160  # resize the image, if the input image is too large, training will be very slow.
	#IMG_H = 120
	startTime = time.time()
	saveName = strftime("%Y-%m-%d %H:%M:%S", gmtime())
	# you need to change the directories to yours.
	train_dir = '/home/shi/Documents/PR970/Leaves/'
	#train_dir = '/home/shi/Documents/PR970/LeavesBW/' #binary image
	#train_dir = '/home/shi/Documents/PR970/LeavesEdges/' #Edges image
	logs_train_dir = '/home/shi/Documents/PR970/Logflavia/Train'
	logs_train_dir = os.path.join(logs_train_dir,saveName)
	
	logs_val_dir = '/home/shi/Documents/PR970/Logflavia/Test'
	logs_val_dir = os.path.join(logs_val_dir,saveName)
	#print "start training...."

	train, train_label, val, val_label = get_files(train_dir, RATIO)
	train_batch, train_label_batch = get_batch(train,
												train_label,
												IMG_W,
												IMG_H,
												BATCH_SIZE, 
												CAPACITY)
	val_batch, val_label_batch = get_batch(val,
											val_label,
											IMG_W,
											IMG_H,
											BATCH_SIZE, 
											CAPACITY)
	
	logits = inference(train_batch, BATCH_SIZE, N_CLASSES)
	loss = losses(logits, train_label_batch)        
	train_op = trainning(loss, learning_rate)
	acc = evaluation(logits, train_label_batch)
	
	x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
	y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE])

	with tf.Session() as sess:
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess= sess, coord=coord)
		
		summary_op = tf.summary.merge_all()        
		train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
		val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
	
		try:
			for step in np.arange(MAX_STEP):
				if coord.should_stop():
						break
				
				tra_images,tra_labels = sess.run([train_batch, train_label_batch])
				_, tra_loss, tra_acc = sess.run([train_op, loss, acc],
												feed_dict={x:tra_images, y_:tra_labels})
				if step % 50 == 0:
					print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
					summary_str = sess.run(summary_op)
					train_writer.add_summary(summary_str, step)
					
				if step % 200 == 0 or (step + 1) == MAX_STEP:
					val_images, val_labels = sess.run([val_batch, val_label_batch])
					val_loss, val_acc = sess.run([loss, acc], 
												 feed_dict={x:val_images, y_:val_labels})
					print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc*100.0))
					summary_str = sess.run(summary_op)
					val_writer.add_summary(summary_str, step)  
									
				if step % 2000 == 0 or (step + 1) == MAX_STEP:
					checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
					saver.save(sess, checkpoint_path, global_step=step)
					
		except tf.errors.OutOfRangeError:
			print('Done training -- epoch limit reached')
		finally:
			coord.request_stop()           
		coord.join(threads)
		endTime = time.time()
		print "total running time: "
		print endTime-startTime

#if __name__ == "__main__":
	#IMG_W,IMG_H
	#(320, 240), 1216s
	# sizeList = [(160,120),(130,98),(100,75),(80,60)]
	# for size in sizeList:
	# 	print size


#run_training(320, 240), #1216s   2017-04-18 21:33:20
#run_training(260,195)    #1100       2017-04-18 22:58:24
#run_training(200,150)    #1014s  2017-04-18 22:39:00
#run_training(160,120) #982s      2017-04-18 22:04:24
#run_training(130,98) #966s       2017-04-18 22:21:21


