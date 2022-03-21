'''
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with
6000 images per class. There are 50000 training images and 10000 test images.
The classes contain: airplane, automobile, bird, cat, deer, dog, frog, horse,
ship and truck
'''

'''
define a function for getting cifar10 images and preprocess the data
'''
#define how many images we need for train,valid,test and dev
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    
    # loading cifar10 dataset with keras library
    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test)= cifar10.load_data()
    print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))

    '''
    divide the dataset to train,validation,test,dev
    '''
    import numpy as np
    
    #defining the range of validation data, We use xtrain for validation data
    #it starts from 49000th image to 50000th image
    mask = list(range(num_training,num_training+num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]

    #defining the range of training data, We use xtrain for training data
    #it starts from 1th image to 49000th image            
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    #defining the range of test data, We use xtest for test data
    #it starts from 1th image to 1000th image 
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    #defining the range of training data, We use xtrain for dev data
    #we choose the number of dev data randomly between 49000 train dataset   
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    '''
    change the 32*32*3 image 3d array into 1d 3072 array
    (reshape the image data into rows)
    '''
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    '''
    normalize the data: subtract the mean image
    '''
    #It is the mean on the column
    mean_image = np.mean(X_train, axis = 0)
    mean_image= mean_image.astype('float32')
    
    X_train= X_train.astype('float32')
    X_train -= mean_image
    
    X_val= X_val.astype('float32')
    X_val -= mean_image
    
    X_test= X_test.astype('float32')
    X_test -= mean_image
    
    X_dev= X_dev.astype('float32')
    X_dev -= mean_image

    '''
    add  bias dimension and transform into columns
    '''
    # with hstack add arrays of 1 comes from np.one as bias to xtrain
    X_train = np.hstack(
    # with np.ones make an array of 1s equals to the number of images      
        [X_train, np.ones((X_train.shape[0], 1))]
        )
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('dev data shape: ', X_dev.shape)
print('dev labels shape: ', y_dev.shape)
