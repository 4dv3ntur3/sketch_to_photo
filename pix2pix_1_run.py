from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot


# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = Activation('swish')(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = Activation('swish')(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = Activation('swish')(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = Activation('swish')(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = Activation('swish')(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model
 
# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g
 
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g
 
# define the standalone generator model
def define_generator(image_shape=(256,256,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g) 

	# define model
	model = Model(in_image, out_image)
	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
	return model
 
# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
 

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y
 
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)

	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0

	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
 
# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=1, n_batch=32):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs

	print("iterations: ", n_steps)

	d_loss1_list = []
	d_loss2_list = []
	g_loss_list = []

	# manually enumerate epochs
	for i in range(n_steps):
    		
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)

		# 
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		d_loss1_list.append(d_loss1)

		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		d_loss2_list.append(d_loss2)

		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		g_loss_list.append(g_loss)

		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))

		# summarize model performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)


	#loss plot
	import matplotlib.pyplot as plt

	x_axis = range(0, n_steps)
	fig, ax = plt.subplots()
	ax.plot(x_axis, d_loss1_list, label="d_loss1")
	ax.plot(x_axis, d_loss2_list, label="d_loss2")

	ax.legend()
	plt.ylabel("Loss")
	plt.xlabel("Iteration")
	plt.title("GAN Loss")
	plt.show()

	fig,ax = plt.subplots()
	ax.plot(x_axis, g_loss_list, label="g_loss")

	ax.legend()
	plt.ylabel("Loss")
	plt.xlabel("Iteration")
	plt.title("GAN Loss")
	plt.show()


# load image data
dataset = load_real_samples('./data/berry_bear.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# Loaded (4164, 256, 256, 3) (4164, 256, 256, 3)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset)







# print("============== 판별자 ================")
# d_model.summary()
# print("output shape: ", d_model.output_shape[1]) #output shape:  (None, 16, 16, 1)

# print("============== 생성기 ================")
# g_model.summary()

# print("=============== GAN =================")
# gan_model.summary()













'''
============== 생성기 ================
Model: "functional_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_3 (InputLayer)            [(None, 256, 256, 3) 0
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 128, 128, 64) 3136        input_3[0][0]
__________________________________________________________________________________________________
leaky_re_lu_5 (LeakyReLU)       (None, 128, 128, 64) 0           conv2d_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 64, 64, 128)  131200      leaky_re_lu_5[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 64, 64, 128)  512         conv2d_7[0][0]
__________________________________________________________________________________________________
leaky_re_lu_6 (LeakyReLU)       (None, 64, 64, 128)  0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 32, 256)  524544      leaky_re_lu_6[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 32, 32, 256)  1024        conv2d_8[0][0]
__________________________________________________________________________________________________
leaky_re_lu_7 (LeakyReLU)       (None, 32, 32, 256)  0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 16, 16, 512)  2097664     leaky_re_lu_7[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 16, 16, 512)  2048        conv2d_9[0][0]
__________________________________________________________________________________________________
leaky_re_lu_8 (LeakyReLU)       (None, 16, 16, 512)  0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 8, 8, 512)    4194816     leaky_re_lu_8[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 8, 8, 512)    2048        conv2d_10[0][0]
__________________________________________________________________________________________________
leaky_re_lu_9 (LeakyReLU)       (None, 8, 8, 512)    0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 4, 4, 512)    4194816     leaky_re_lu_9[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 4, 4, 512)    2048        conv2d_11[0][0]
__________________________________________________________________________________________________
leaky_re_lu_10 (LeakyReLU)      (None, 4, 4, 512)    0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 2, 2, 512)    4194816     leaky_re_lu_10[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 2, 2, 512)    2048        conv2d_12[0][0]
__________________________________________________________________________________________________
leaky_re_lu_11 (LeakyReLU)      (None, 2, 2, 512)    0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 1, 1, 512)    4194816     leaky_re_lu_11[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 1, 1, 512)    0           conv2d_13[0][0]
__________________________________________________________________________________________________
conv2d_transpose (Conv2DTranspo (None, 2, 2, 512)    4194816     activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 2, 2, 512)    2048        conv2d_transpose[0][0]
__________________________________________________________________________________________________
dropout (Dropout)               (None, 2, 2, 512)    0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 2, 2, 1024)   0           dropout[0][0]
                                                                 leaky_re_lu_11[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 2, 2, 1024)   0           concatenate_1[0][0]
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 4, 4, 512)    8389120     activation_2[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 4, 4, 512)    2048        conv2d_transpose_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 4, 4, 512)    0           batch_normalization_11[0][0]
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 4, 4, 1024)   0           dropout_1[0][0]
                                                                 leaky_re_lu_10[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 4, 4, 1024)   0           concatenate_2[0][0]
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 8, 8, 512)    8389120     activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 8, 8, 512)    2048        conv2d_transpose_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 8, 8, 512)    0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 8, 8, 1024)   0           dropout_2[0][0]
                                                                 leaky_re_lu_9[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 8, 8, 1024)   0           concatenate_3[0][0]
__________________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTrans (None, 16, 16, 512)  8389120     activation_4[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 16, 16, 512)  2048        conv2d_transpose_3[0][0]
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 16, 16, 1024) 0           batch_normalization_13[0][0]
                                                                 leaky_re_lu_8[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 16, 16, 1024) 0           concatenate_4[0][0]
__________________________________________________________________________________________________
conv2d_transpose_4 (Conv2DTrans (None, 32, 32, 256)  4194560     activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 32, 32, 256)  1024        conv2d_transpose_4[0][0]
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 32, 32, 512)  0           batch_normalization_14[0][0]
                                                                 leaky_re_lu_7[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 32, 512)  0           concatenate_5[0][0]
__________________________________________________________________________________________________
conv2d_transpose_5 (Conv2DTrans (None, 64, 64, 128)  1048704     activation_6[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 64, 64, 128)  512         conv2d_transpose_5[0][0]
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 64, 64, 256)  0           batch_normalization_15[0][0]
                                                                 leaky_re_lu_6[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 64, 64, 256)  0           concatenate_6[0][0]
__________________________________________________________________________________________________
conv2d_transpose_6 (Conv2DTrans (None, 128, 128, 64) 262208      activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 128, 128, 64) 256         conv2d_transpose_6[0][0]
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 128, 128, 128 0           batch_normalization_16[0][0]
                                                                 leaky_re_lu_5[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 128, 128, 128 0           concatenate_7[0][0]
__________________________________________________________________________________________________
conv2d_transpose_7 (Conv2DTrans (None, 256, 256, 3)  6147        activation_8[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 256, 256, 3)  0           conv2d_transpose_7[0][0]
==================================================================================================
'''
