from mnist import MNIST

mdata = MNIST('/home/evgenyorlov1/Echo-State-Network/datasets/MNIST')

train_images, train_labels = mdata.load_training()

test_images, test_labels = mdata.load_testing()

print 'type train: {0}'.format(len(test_images[0]))