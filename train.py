import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
# physical_devices=tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0],True)

# import the necessary packages
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from model.fashionnet import FashionNet
from imutils import paths

import numpy as np
import argparse
import random
import pickle
import cv2
import os
import collections


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--categorybin", required=True,
	help="path to output category label binarizer")
ap.add_argument("-c", "--colorbin", required=True,
	help="path to output color label binarizer")

args = vars(ap.parse_args())

# args={"dataset":"dataset","model":"output/fashion_model.h5","categorybin":"output/category_lb.pickle",
# 	  "colorbin":"output/color_lb.pickle","plot":"output"}

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 1
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (100, 100, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data, clothing category labels (i.e., shirts, jeans,
# dresses, etc.) along with the color labels (i.e., red, blue, etc.)
data = []
categoryLabels = []
colorLabels = []

class0Labels = []
class1Labels = []
class2Labels = []
class3Labels = []
class4Labels = []
class5Labels = []
class6Labels = []
class7Labels = []



# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	if imagePath.find('#') >= 0:
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = img_to_array(image)
		data.append(image)
		# extract the clothing color and category from the path and
		# update the respective lists
		(class0, class1, class2, class3, class4, class5, class6, class7) = imagePath.split(os.path.sep)[-1].split('#')[
																			   1].split('_')[:-1]
		class0Labels.append(class0)
		class1Labels.append(class1)
		class2Labels.append(class2)
		class3Labels.append(class3)
		class4Labels.append(class4)
		class5Labels.append(class5)
		class6Labels.append(class6)
		class7Labels.append(class7)

# scale the raw pixel intensities to the range [0, 1] and convert to
# a NumPy array
data = np.array(data, dtype="float") / 255.0
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
    len(imagePaths), data.nbytes / (1024 * 1000.0)))

# convert the label lists to NumPy arrays prior to binarization
categoryLabels = np.array(categoryLabels)
colorLabels = np.array(colorLabels)

class0Labels = np.array(class0Labels)
class1Labels = np.array(class1Labels)
class2Labels = np.array(class2Labels)
class3Labels = np.array(class3Labels)
class4Labels = np.array(class4Labels)
class5Labels = np.array(class5Labels)
class6Labels = np.array(class6Labels)
class7Labels = np.array(class7Labels)

# binarize both sets of labels
print("[INFO] binarizing labels...")

class0LB = LabelBinarizer()
class1LB = LabelBinarizer()
class2LB = LabelBinarizer()
class3LB = LabelBinarizer()
class4LB = LabelBinarizer()
class5LB = LabelBinarizer()
class6LB = LabelBinarizer()
class7LB = LabelBinarizer()
class0Labels = class0LB.fit_transform(class0Labels)
class1Labels = class1LB.fit_transform(class1Labels)
class2Labels = class2LB.fit_transform(class2Labels)
class3Labels = class3LB.fit_transform(class3Labels)
class4Labels = class4LB.fit_transform(class4Labels)
class5Labels = class5LB.fit_transform(class5Labels)
class6Labels = class6LB.fit_transform(class6Labels)
class7Labels = class7LB.fit_transform(class7Labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, class0Labels, class1Labels, class2Labels, class3Labels, class4Labels, class5Labels, class6Labels, class7Labels,
	test_size=0.2, random_state=42)
(trainX, testX, trainClass0Y, testClass0Y, trainClass1Y, testClass1Y, trainClass2Y, testClass2Y,
 trainClass3Y, testClass3Y, trainClass4Y, testClass4Y,
 trainClass5Y, testClass5Y, trainClass6Y, testClass6Y, trainClass7Y, testClass7Y) = split

# initialize our FashionNet multi-output network
model = FashionNet.build(100, 100,
						 numClass0=len(class0LB.classes_),
						 numClass1=len(class1LB.classes_),
						 numClass2=len(class2LB.classes_),
						 numClass3=len(class3LB.classes_),
						 numClass4=len(class4LB.classes_),
						 numClass5=len(class5LB.classes_),
						 numClass6=len(class6LB.classes_),
						 numClass7=len(class7LB.classes_),
						 finalAct="softmax")

# plot_model(model)

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {}
losses['class0_output'] = 'sparse_categorical_crossentropy'
losses['class1_output'] = 'sparse_categorical_crossentropy'
losses['class2_output'] = 'sparse_categorical_crossentropy'
losses['class3_output'] = 'sparse_categorical_crossentropy'
losses['class4_output'] = 'sparse_categorical_crossentropy'
losses['class5_output'] = 'categorical_crossentropy'
losses['class6_output'] = 'categorical_crossentropy'
losses['class7_output'] = 'categorical_crossentropy'

lossWeights = {}
lossWeights['class0_output'] = 1.0
lossWeights['class1_output'] = 1.0
lossWeights['class2_output'] = 1.0
lossWeights['class3_output'] = 1.0
lossWeights['class4_output'] = 1.0
lossWeights['class5_output'] = 1.0
lossWeights['class6_output'] = 1.0
lossWeights['class7_output'] = 1.0

trainDic = {}
trainDic['class0_output'] = trainClass0Y
trainDic['class1_output'] = trainClass1Y
trainDic['class2_output'] = trainClass2Y
trainDic['class3_output'] = trainClass3Y
trainDic['class4_output'] = trainClass4Y
trainDic['class5_output'] = trainClass5Y
trainDic['class6_output'] = trainClass6Y
trainDic['class7_output'] = trainClass7Y

validataDic = {}
validataDic['class0_output'] = testClass0Y
validataDic['class1_output'] = testClass1Y
validataDic['class2_output'] = testClass2Y
validataDic['class3_output'] = testClass3Y
validataDic['class4_output'] = testClass4Y
validataDic['class5_output'] = testClass5Y
validataDic['class6_output'] = testClass6Y
validataDic['class7_output'] = testClass7Y

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])

# train the network to perform multi-output classification
H = model.fit(trainX, trainDic, validation_data=(testX, validataDic), epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the category binarizer to disk
print("[INFO] serializing class0 label binarizer...")
f = open('output/class0bin.pickle', "wb")
f.write(pickle.dumps(class0LB))
f.close()

print("[INFO] serializing class1 label binarizer...")
f = open('output/class1bin.pickle', "wb")
f.write(pickle.dumps(class1LB))
f.close()

print("[INFO] serializing class2 label binarizer...")
f = open('output/class2bin.pickle', "wb")
f.write(pickle.dumps(class2LB))
f.close()

print("[INFO] serializing class3 label binarizer...")
f = open('output/class3bin.pickle', "wb")
f.write(pickle.dumps(class3LB))
f.close()

print("[INFO] serializing class4 label binarizer...")
f = open('output/class4bin.pickle', "wb")
f.write(pickle.dumps(class4LB))
f.close()

print("[INFO] serializing class5 label binarizer...")
f = open('output/class5bin.pickle', "wb")
f.write(pickle.dumps(class5LB))
f.close()

print("[INFO] serializing class6 label binarizer...")
f = open('output/class6bin.pickle', "wb")
f.write(pickle.dumps(class6LB))
f.close()

print("[INFO] serializing class7 label binarizer...")
f = open('output/class7bin.pickle', "wb")
f.write(pickle.dumps(class7LB))
f.close()



# # plot the total loss, category loss, and color loss
# lossNames = ["loss", "category_output_loss", "color_output_loss"]
# plt.style.use("ggplot")
# (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
#
# # loop over the loss names
# for (i, l) in enumerate(lossNames):
# 	# plot the loss for both the training and validation data
# 	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
# 	ax[i].set_title(title)
# 	ax[i].set_xlabel("Epoch #")
# 	ax[i].set_ylabel("Loss")
# 	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
# 	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
# 		label="val_" + l)
# 	ax[i].legend()
#
# # save the losses figure
# plt.tight_layout()
# plt.savefig("{}_losses.png".format(args["plot"]))
# plt.close()
#
# create a new figure for the accuracies
# accuracyNames = ["category_output_accuracy", "color_output_accuracy"]
# plt.style.use("ggplot")
# (fig, ax) = plt.subplots(2, 1, figsize=(8, 8))
#
# # loop over the accuracy names
# for (i, l) in enumerate(accuracyNames):
# 	# plot the loss for both the training and validation data
# 	ax[i].set_title("Accuracy for {}".format(l))
# 	ax[i].set_xlabel("Epoch #")
# 	ax[i].set_ylabel("Accuracy")
# 	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
# 	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
# 		label="val_" + l)
# 	ax[i].legend()
#
# # save the accuracies figure
# plt.tight_layout()
# plt.savefig("{}_accs.png".format(args["plot"]))
# plt.close()