import numpy as np

n_samples=3

# select a batch of random samples, returns images and target
# def generate_real_samples_2(dataset, n_samples, patch_shape, iter):

# unpack dataset
# sketch, photo = dataset
ix = np.arange(10)
# ix = ix.tolist()
iter = 1
ix = ix[iter*n_samples:(iter+1)*n_samples]

print("new: ", ix)
	
	# # retrieve selected images
	# X1, X2 = sketch[ix], photo[ix]
	# # generate 'real' class labels (1)
	# y = np.ones((n_samples, patch_shape, patch_shape, 1))
	# return [X1, X2], y


# select a batch of random samples, returns images and target
# def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
# trainA, trainB = dataset
# choose random instances
ix = np.random.randint(0, 10, n_samples)

print("1: ", ix)
# retrieve selected images
# X1, X2 = trainA[ix], trainB[ix]
# # generate 'real' class labels (1)
# y = np.ones((n_samples, patch_shape, patch_shape, 1))
# return [X1, X2], y


