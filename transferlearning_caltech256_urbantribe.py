import os
import tarfile
import numpy as np
from scipy import misc
import keras
from keras.layers import Dense
from keras.models import Model
from keras.models import Sequential
from keras.applications import VGG16
import matplotlib.pyplot as plt


class ExtractFile(object):
	def __init__(self,filename):
		'''
		Initialize object to extract images from tar file located at root
		:param filename: filename of tar file
		:return: NONE
		'''
		self.root = os.path.splitext(os.path.splitext(filename)[0])[0]
		self.filename = filename

	def uncompress(self):
		'''
		Uncompress tar file if uncompressed file is not available
		:return: NONE
		'''
		if os.path.isdir(self.root):
			print('Images already extracted ')
		else:
			print('Extracting data...')
			tar = tarfile.open(self.filename)
			tar.extractall(path="data")
			tar.close()

	def get_root(self):
		'''
		Get root directory where the images are categorically stored
		:return:
		'''
		return self.root


class Preprocess(object):
	def __init__(self, image_size):
		self.image_size = image_size
		self.channel = image_size[2]

	@staticmethod
	def shuffle(data, labels):
		'''
		Randomly shuffle data for better performance
		:param data: numpy array of image dataset
		:param labels: numpy array of image labels
		:return: shuffled image and labels
		'''
		permutation = np.random.permutation(data.shape[0])
		labels = labels[permutation]
		data = data[permutation]
		return data, labels

	@staticmethod
	def normalize_data(data):
		'''
		Normalize data by subtracting mean and dividing by standard deviation
		:param data: numpy array of image dataset
		:return: normalized image dataset
		'''
		data -= data.mean(axis=0)
		return data/np.std(data, axis=0)

	@staticmethod
	def get_image_folders(root_path):
		'''
		Get list of directories with images
		:param root_path: root path containing image subdirectories
		:return: list of image folders
		'''
		image_folders = [os.path.join(root_path, d) for d in sorted(os.listdir(root_path)) if os.path.isdir(
			os.path.join(root_path, d))]
		return image_folders

	def process_all_images(self, directories, images_per_category):
		'''
		From each directory, extact images, randomly pick specified set of images, resize them and load into array
		:param directories: list of subdirectories with images
		:param images_per_category: number of images per category to be loaded
		:return: numpy array with data and labels
		'''
		count = 0
		label = 0
		labels = np.array([])
		for directory in directories:
			images = np.array([f for f in os.listdir(directory) if f.lower().endswith('.jpg')])[:images_per_category]
			for f in images:
				count += 1
				labels = np.append(labels, label)
			label += 1

		# Initialize empty data array
		data = np.zeros((count, self.image_size[0], self.image_size[1], self.channel))
		count = 0

		for directory in directories:
			images = np.array([f for f in os.listdir(directory) if f.lower().endswith('.jpg')])
			# Randomly pick a fixed set of images from each category
			permutation = np.random.permutation(len(images))[:images_per_category]

			for f in images[permutation]:
				data[count] = np.reshape(misc.imresize(
					misc.imread(os.path.join(directory, f), mode='RGB'),self.image_size),
					(-1, self.image_size[0], self.image_size[1], self.channel)
				)
				count += 1

		# Assign labels to each image
		labels = (np.arange(np.min(labels), np.max(labels)+1) == labels[:, None]).astype(float)
		return data, labels


class DisplayStatistics(object):

	def plot_history(self, model_history):
		'''
		Plot statics of model performance based on history data
		:param model_history: list of model history objects
		:return: NONE
		'''
		self.plot_loss_iterations(model_history)
		self.plot_acccuracy_iterations(model_history)

	@staticmethod
	def plot_loss_iterations(model_history):

		x = [i for i in xrange(len(model_history.history['loss']))]
		y1 = model_history.history['loss']
		y2 = model_history.history['val_loss']
		image_per_class = ['2','4','8','16']
		color = ['r','g','b','c']
		for i in xrange(0,4):
			plt.plot(x, model_history[i].history['loss'], color[i],
			         label='train loss' + image_per_class[i] +'image per class')
			plt.plot(x, model_history[i].history['val_loss'], color[i],
			         label='test loss'+  image_per_class[i] +'image per class', ls='dashdot')
		plt.xlabel('iterations')
		plt.ylabel('loss value')
		plt.title('loss vs iterations')
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	@staticmethod
	def plot_acccuracy_iterations(model_history):
		x = [i for i in xrange(len(model_history.history['loss']))]
		image_per_class = ['2','4','8','16']
		color = ['r','g','b','c']
		for i in xrange(0,4):
			plt.plot(x, model_history[i].history['acc'], color[i],
			         label ='train accuracy' + image_per_class[i] + 'image per class')
			plt.plot(x, model_history[i].history['val_acc'], color[i],
			         label='test accuracy' + image_per_class[i] + 'image per class', ls='dashdot')
		plt.xlabel('iterations')
		plt.ylabel('accuracy')
		plt.title('iterations vs accuracy')
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


class TransferLearning(object):

	@staticmethod
	def get_vgg_model(layers_to_remove):
		'''
		Generate VGG16 model with speficied number of layers removed to implement with pre-trained weights
		:param output_dim:
		:return: VGG16 pre-trained model with last two layers replaced with Softmax of 256 dimensions
		'''
		vgg_model = keras.applications.VGG16(weights='imagenet', include_top=True)

		# Use all layers except the last two
		vgg_out = vgg_model.layers[-layers_to_remove].output

		# Build transfer learning model with predefined layers
		tl_model = Model(input=vgg_model.input, output=vgg_out)

		# Freeze all layers of VGG16 and Compile the model
		for layer_idx in range(len(tl_model.layers)):
			tl_model.layers[layer_idx].trainable = False

		# Compile the configured model
		tl_model.compile(
			optimizer='rmsprop',
			loss='categorical_crossentropy',
			metrics=['accuracy'])

		# Confirm the model is appropriate
		print tl_model.summary()

		return tl_model

	@staticmethod
	def softmax_model(vgg16_output, output_dim):
		soft_model = Sequential()
		soft_model.add(Dense(output_dim=output_dim, activation='softmax', input_dim=vgg16_output.shape[1]))
		soft_model.compile(
			optimizer='rmsprop',
			loss='categorical_crossentropy',
			metrics=['accuracy']
		)
		print soft_model.summary()
		return soft_model


class GetData(object):

	def __init__(self, root_path, images_per_category, to_extract):
		self.filename = root_path
		self.image_size = [224, 224, 3]
		self.num_images_per_category = images_per_category
		self.to_extract = to_extract

	def __extract_file(self):
		extracter = ExtractFile(self.filename)
		extracter.uncompress()
		print('Uncompression Complete !')
		return extracter.get_root()

	def __preprocess_images(self, root_path):

		preprocessor = Preprocess(self.image_size)

		# Get list of image folders
		image_folders = preprocessor.get_image_folders(root_path)

		# Convert all images into tuple of numpy data array and labels
		data, labels = preprocessor.process_all_images(image_folders, self.num_images_per_category)

		# Normalize data and shuffle the dataset
		data = preprocessor.normalize_data(data)
		data, labels = preprocessor.shuffle(data, labels)

		return data, labels

	def visualize_layers(self, img_per_category):
		'''
		Visualize intermediate layers in the VGG Convolutional Neural Network
		:param img_per_category:
		:return: NONE
		'''
		layers = ['block1_conv1', 'block2_conv2']
		layers_to_remove = 2

		root, data, caltech_label = self.get_processed_data()
		single_data = np.array([data[0]])

		tl = TransferLearning()
		model = tl.get_vgg_model(layers_to_remove)

		for layer_name in layers:
			intermediate_layer_model = Model(input=model.input, output=model.get_layer(layer_name).output)
			intermediate_output = intermediate_layer_model.predict(single_data)
			layer_outputs = intermediate_output[:][:][:][0]

			plt.figure(figsize=(15,15))
			for i in range(0,64):
				plt.subplot(8,8,i+1)
				plt.imshow(layer_outputs[i].T)
			plt.show()

	@staticmethod
	def split_train_test(data, label):
		return data[:int(0.8*data.shape[0])],\
		       label[:int(0.8*label.shape[0])],\
		       data[int(0.2*data.shape[0]):],\
		       label[int(0.2*label.shape[0]):]

	def get_processed_data(self):

		# Uncompress tar file and load images
		if self.to_extract == True:
			root = self.__extract_file()
		else:
			root = self.filename

		# Preprocess images and load as numpy array for training
		data, labels = self.__preprocess_images(root)

		return root,data, labels


def transferlearn_caltech256():

	num_of_classes = 257
	num_of_epochs = 25
	batch_size = 32
	validation_split = 0.3
	to_extract = True
	layers_to_remove = [2]
	num_images_per_category = [2]
	display_history = DisplayStatistics()

	# Array to store Train Validation Statistics for Model fitting
	caltech_model_fit_history = []

	for img_per_category in num_images_per_category:

		#Extract and process dataset with specified images per category
		caltech256 = GetData(
			'data/256_ObjectCategories.tar',
			img_per_category,
			to_extract
		)
		root, caltech_data, caltech_label = caltech256.get_processed_data()
		tl = TransferLearning()

		# For each model with layers removed at tail end, forward propagate input
		# through ImageNet and store output of network in numpy array file
		for to_remove in layers_to_remove:

			# Get VGG16 model wit layers removed
			tl_model = tl.get_vgg_model(to_remove)
			numpy_array_file = "caltech256_" + str(to_remove) + ".npy"

			if os.path.isfile(numpy_array_file):
				print str(numpy_array_file) + " exists -- Skipping forward propagation"
				continue

			print "Propagating CalTech data through VGG16 NN with " + str(to_remove) + " layers removed"
			# Get output for dataset
			vgg16_output = tl_model.predict(caltech_data)
			print "Forward Propagation Complete"

			# Store in file for future processing
			# TODO: Comment out this code block once complete. It is a one time action

			np.save(numpy_array_file,vgg16_output)
			print "Stored numpy array in file : " + str(numpy_array_file)

		for to_remove in layers_to_remove:

			# Load forward propagated output for each model
			numpy_array_file = "caltech256_" + str(to_remove) + ".npy"
			loaded_vgg16_output = np.load(numpy_array_file)
			print "Loaded forward propagated output from VGG16 for " + str(to_remove) + " removed network"

			# Train Softmax model with VGG16's output as input
			vgg16_tl_model = tl.softmax_model(loaded_vgg16_output, num_of_classes)
			m_history = vgg16_tl_model.fit(
				loaded_vgg16_output,
				caltech_label,
				nb_epoch=num_of_epochs,
				batch_size=batch_size,
				validation_split=validation_split)

			caltech_model_fit_history.append(m_history)

	display_history.plot_history(caltech_model_fit_history)


def transferlearn_urbantribe():

	num_of_classes = 11
	num_of_epochs = 10
	batch_size = 32
	to_extract = False
	layers_to_remove = [2]
	num_images_per_category = [2,4,8,16]
	display_history = DisplayStatistics()

	# Array to store Train Validation Statistics for Model fitting
	caltech_model_fit_history = []

	for img_per_category in num_images_per_category:
		# Extract and process dataset with specified images per category
		urban = GetData(
			'data/pictures_all',
			img_per_category,
			to_extract)
		root, urban_data, urban_label = urban.get_processed_data()
		urban_train_data, urban_train_label, urban_test_data, urban_test_label = urban.split_train_test(urban_data,
																										urban_label)
		tl = TransferLearning()

		for to_remove in layers_to_remove:
			# Get VGG16 model wit layers removed
			tl_model = tl.get_vgg_model(to_remove)
			urban_train_data_file = "urbantribe_train_data_"  + str(img_per_category) + "_imgs.npy"
			urban_train_label_file = "urbantribe_train_label_"  + str(img_per_category) + "_imgs.npy"
			urban_test_data_file = "urbantribe_test_data_"  + str(img_per_category) + "_imgs.npy"
			urban_test_label_file = "urbantribe_test_label_"  + str(img_per_category) + "_imgs.npy"

			if os.path.isfile(urban_train_data_file) and os.path.isfile(urban_test_data_file) and \
					os.path.isfile(urban_train_label_file) and os.path.isfile(urban_test_label_file):
				print str(urban_train_data_file) + " exists -- Skipping forward propagation"
				continue
			print "Propagating UrbanTribe data through VGG16 NN with " + str(to_remove) + " layers removed"
			# Get output for dataset
			urban_vgg_train_output = tl_model.predict(urban_train_data)
			urban_vgg_test_output = tl_model.predict(urban_test_data)
			print "Forward Propagation Complete"

			np.save(urban_train_data_file, urban_vgg_train_output)
			np.save(urban_test_data_file, urban_vgg_test_output)
			np.save(urban_train_label_file, urban_train_label)
			np.save(urban_test_label_file, urban_test_label)
			print "Stored numpy arrays in files.."

		for to_remove in layers_to_remove:
			urban_train_data_file = "urbantribe_train_data_"  + str(img_per_category) + "_imgs.npy"
			urban_train_label_file = "urbantribe_train_label_"  + str(img_per_category) + "_imgs.npy"
			urban_test_data_file = "urbantribe_test_data_"  + str(img_per_category) + "_imgs.npy"
			urban_test_label_file = "urbantribe_test_label_"  + str(img_per_category) + "_imgs.npy"

			urban_train_data = np.load(urban_train_data_file)
			urban_train_label = np.load(urban_train_label_file)
			urban_test_data = np.load(urban_test_data_file)
			urban_test_label = np.load(urban_test_label_file)

			print "Loaded forward propagated output from VGG16 for " + str(to_remove) + " removed network"

			# Train Softmax model with VGG16's output as input
			vgg16_tl_model = tl.softmax_model(urban_train_data, num_of_classes)
			m_history = vgg16_tl_model.fit(
				urban_train_data,
				urban_train_label,
				nb_epoch=num_of_epochs,
				batch_size=batch_size,
				validation_data=(urban_test_data, urban_test_label))

			caltech_model_fit_history.append(m_history)
		display_history.plot_history(caltech_model_fit_history)

