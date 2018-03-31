import random
import sys
import numpy as np
from scipy.special import expit
from scipy import misc, stats

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import pickle

def get_data(img_width = 30, img_height = 30,
             file_path = 'Data_Train', class_names = None,
             validation_ratio = 0.2, nb_samples = [1000,1000,1000],
             data_augmentation=False):
    
    if class_names is None: # testing, no answer
        img_path = [str(a)+'.bmp' for a in range(1, 601)]
        print(img_path)
        df = pd.DataFrame(index = range(len(img_path)), columns = range(img_height*img_width))
        for file, i in zip(img_path, range(len(img_path))):
            img = misc.imread(file_path + '/' + file).flatten()
            df.iloc[i] = img
        return df
        
    train_df = pd.DataFrame(columns = range(img_height*img_width))
    train_df['class'] = 0
    augmentation_df = pd.DataFrame(columns = range(img_height*img_width))
    augmentation_df['class'] = 0
    for i in range(len(class_names)):
        path = file_path + '/' + class_names[i] + '/'
        print(path)
        class_img_files = os.listdir(path)[:nb_samples[i]]
        
        df = pd.DataFrame(index = range(len(class_img_files)), columns = range(img_height*img_width))
        aug_df = pd.DataFrame(columns = range(img_height*img_width))
        for file, j in zip(class_img_files, range(len(class_img_files))):
            img_path = path + file
            img = misc.imread(img_path).flatten()
            df.iloc[j] = img
            if data_augmentation:
                img = img.reshape((img_height, img_width))
                # rotate
                angle = [random.randint(10, 20), random.randint(-20, -10)][random.randint(0,1)]
                img1 = misc.imrotate(img, angle)
                img1 = img1.flatten()        
                aug_df = aug_df.append(pd.Series(img1), ignore_index=True)
                # flip
                img2 = np.fliplr(img)
                img2 = img2.flatten()
                aug_df = aug_df.append(pd.Series(img2), ignore_index=True)
        df['class'] = class_names[i]
        aug_df['class'] = class_names[i]
        train_df = train_df.append(df)
        augmentation_df = augmentation_df.append(aug_df)
        
    np.random.seed(0)
    train_df = train_df.sample(frac=1)
    train_df, val_df = train_df.iloc[:int(len(train_df)*(1-validation_ratio))], train_df.iloc[int(len(train_df)*(1-validation_ratio)):]
    train_df = train_df.append(augmentation_df)
    
    train_class = train_df['class']
    train_class.index = range(len(train_class))
    train_df = train_df.drop('class', axis=1)
    train_df.index = range(len(train_df))
    val_class = val_df['class']
    val_class.index = range(len(val_class))
    val_df = val_df.drop('class', axis=1)
    val_df.index = range(len(val_df))
    
    return train_df, train_class, val_df, val_class

def preprocessing(train_df, train_class, val_df, val_class, class_names = None, pca=True):
    
    class_to_one_hot = {}
    for i, j in zip(np.sort(class_names), range(len(class_names))):
        one_hot = np.zeros((len(class_names), 1))
        one_hot[j] = 1
        class_to_one_hot[i] = one_hot
    if pca:
        pca = PCA(n_components=2).fit(scale(train_df))
        pca_train = pca.fit_transform(scale(train_df))
        pca_val = pca.fit_transform(scale(val_df))

        train_inputs = [np.ndarray.astype(row.reshape((2, 1)), np.float64) for row in pca_train]
        val_inputs = [np.ndarray.astype(row.reshape((2, 1)), np.float64) for row in pca_val]
    else:
        train_inputs = [np.ndarray.astype(row.reshape((900, 1)), np.float64) for row in train_df.values]
        val_inputs = [np.ndarray.astype(row.reshape((900, 1)), np.float64) for row in val_df.values]        
    train_results = [class_to_one_hot[i] for i in train_class]
    training_data = zip(train_inputs, train_results)
    
    val_results = [class_to_one_hot[i] for i in val_class]
    validation_data = zip(val_inputs, val_results)
    return list(training_data), list(validation_data)



class NeuronNetwork():
    
    def __init__(self, auto=True):
        """Initialize NN
        
        Args:
            auto (bool): True: you don't have to create output layer
                               we'll add it automatically
                         False: the last layer you added is your output layer # not tested
        
        Returns:
            None
        """
        np.random.seed(0) # for same result
        self.num_layers = 1 # set input layer as layer 0
        self.fitted = False # fit any data yet?
        self.weights = {} # record weights of each layer
        self.biases = {} # record biases of each layer
        self.num_neurons = {} # record number of neurons of each layer
        self.auto = auto # you know it
        self.activation_functions = {'sigmoid':self.sigmoid, 'relu':self.relu, 'tanh':self.tanh} # for clean
        self.f = None # alias of activeation function (bad idea)
        self.seesee = {} # for debug
        self.MAX = None # prevent overflow of sigmoid

    def add_layer(self, num_neurons, initialization='normal', input_size=None):
        """Add layers into NN
        
        Args:
            num_neurons (int): number of neurons of this layer
            initialization (str): 1. normal: normal distribution, zero mean and unit variance
                                  2. scaled_normal: result of 1. multiply sqrt(1/n), n is number of data
                                  3. uniform_random: uniform random from 0 to 1
            input_size (int): only used at first layer, need to specify the size
        Returns:
            None
        """
        np.random.seed(0)
        if self.num_layers == 1: # layer 1, don't know input structure yet
            if input_size == None:
                print('in first layer, you need to specity the input size.')
                return
            self.num_neurons[self.num_layers] = num_neurons
            if initialization == 'normal': # bad
                self.weights[self.num_layers] = np.random.normal(scale=1, 
                                                   size=(num_neurons, input_size))
            elif initialization == 'scaled_normal_relu': # best
                self.weights[self.num_layers] = (np.random.normal(scale=1, 
                                                    size=(num_neurons, input_size))
                                                   *np.sqrt(2/self.num_neurons[1]))
            elif initialization == 'scaled_normal':
                self.weights[self.num_layers] = (np.random.normal(scale=1, 
                                                    size=(num_neurons, input_size))
                                                   /self.num_neurons[1])                
            elif initialization == 'uniform_random': # soso
                self.weights[self.num_layers] = np.random.random(size=(num_neurons, input_size))
            
            self.biases[self.num_layers] = np.zeros(shape=(num_neurons,1))
            print('add layer ', self.num_layers)
            self.num_layers += 1
            
        else:
            if initialization == 'normal':
                self.weights[self.num_layers] = np.random.normal(scale=1, 
                                                                 size=(num_neurons, self.num_neurons[self.num_layers-1]))
            elif initialization == 'scaled_normal_relu':
                self.weights[self.num_layers] = (np.random.normal(scale=1, 
                                                    size=(num_neurons, self.num_neurons[self.num_layers-1]))
                                                   *np.sqrt(2/self.num_neurons[1])) 
            elif initialization == 'scaled_normal':
                self.weights[self.num_layers] = (np.random.normal(scale=1, 
                                                                  size=(num_neurons, self.num_neurons[self.num_layers-1]))
                                                   /self.num_neurons[1])                
            elif initialization == 'uniform_random':
                self.weights[self.num_layers] = np.random.random(size=(num_neurons, self.num_neurons[self.num_layers-1]))
            self.biases[self.num_layers] = np.zeros(shape=(num_neurons, 1))
            self.num_neurons[self.num_layers] = num_neurons
            print('add layer ', self.num_layers)
            self.num_layers += 1
            

    def tanh(self, x, derivative=False):
        pass

    def sigmoid(self, x, derivative=False):
        if not derivative:
            return np.exp(-np.logaddexp(0, -x))
        else:
            return (1-sigmoid(x))*sigmoid(x)
#         if not derivative:
#             return 1.0 / (1.0 + np.exp(-(x+self.MAX)))
#             #return expit(x)
#         else:
#             return (1-sigmoid(x))*sigmoid(x)
#             #return (1-expit(x))*expit(x)
        
    def softmax(self, x):
        MAX = np.max(x)
        print(MAX)
        return np.exp(x-MAX) / np.sum(np.exp(x-MAX))
    
    # leaky relu
    def relu(self, x, derivative=False, leaky=False, alpha=0.01):
        result = []
        if not derivative:
            if leaky:
                results = np.array([alpha*a if a < 0 else a for a in x]).reshape(-1,1)
                print(results)
                return results
            else:
                return (x*(x>0)).reshape(-1,1)
                #return np.array([a*(a>0) for a in x]).reshape(-1,1)
            #return np.apply_along_axis(lambda a: alpha*a if a < 0 else a, 1, x).reshape(-1,1)
        else:
            if leaky:
                results = x*(x>0)/x
                results[np.isnan(results)] = alpha
                return results
                return np.array([alpha if a < 0 else 1 for a in x]).reshape(-1,1)
            else:
                results = x*(x>0)/x
                results[np.isnan(results)] = 0
                return results        
                return np.array([0 if a < 0 else 1 for a in x]).reshape(-1,1)
            #return np.apply_along_axis(lambda a: alpha if a < 0 else 1, 1, x).reshape(-1,1)
    
    def predict(self, x):
        """the prediction of given data
        we don't need softmax to inference
        
        Args:
            x (numpy.ndarray): the data point
        Returns:
            the result of prediction
        """
        if not self.fitted:
            print('not fitted')
            return
        # if we have 3 weights, they are w_1, w_2, w_3
        # which means we have 3-layers (exclude input layer)
        a = x
        ##print('input: \n', a)
        for i in range(1, self.num_layers+1):
            W = self.weights[i]
            #print('w: \n ', W)
            b = self.biases[i]
            #print('b: \n', b)
            z = np.dot(W,a) + b
            #print('z: \n', z)
            a = self.f(z)
            #print('a: \n', a)
        return a
    
    def fit(self, training_data, batch_size=64, epochs=10, activation_function='relu', 
            learning_rate=0.001, decreasing_rate=0.9, test_data=None, verbose=False,
            dropout=None, l2 = None, update_method='SGD', mu=None, v=True): 
        """Fit the data
        
        Args:
            training_data (list): 
                training_data = [sample data points]
                sample data points = (input data point, target of input data point)
            batch_size (int): batch size
            epochs (int): epoch
            activation_function (string): 1. relu
                                          2. sigmoid
                                          3. tanh # not implemented
            learning_rate (float): learning rate
            decreasing_rate (float): after one epoch, learning = learning*decreasing_rate
            test_data (numpy.ndarray): for evaluation
            verbose (bool): for more infomation when training
            dropout (float): the rate of dropout, this value is between 0 and 1 # not implemented
            l2 (float): l2 regularization # not implemented
            update_method (string): 1. SGD, stochastic gradient descent
                                    2. momentum # implemented
                                    3. Adagrad # not implemented
                                    4. RMSprop # not implemented
                                    5. Adam # not implemented
            mu (float): for all momentum-like update methods
            v (bool): for momentum-like methods
        Returns:
            None
        """
        
        self.MAX = max([np.max(data[0]) for data in training]) # prevent overflow of sigmoid
        np.random.seed(0)
        next_mu = mu
        if mu is not None:
            friction_range = 0.99-mu # bad coding style :( 
        self.f = self.activation_functions[activation_function]
        if self.auto and not self.fitted:
            print("did not specify the output layer, but set 'auto', automatically added output layer based on the shape of the target.")
            self.add_layer(training_data[0][1].shape[0])
            self.num_layers -= 1
        self.fitted = True
        print('start training!')
        for i in range(epochs):
            #print(self.weights[2])
            
            np.random.shuffle(training_data) # shuffle the data
            batchs = [training_data[k:k+batch_size] for k in range(0,len(training_data),batch_size)] # create batches
            for batch, j in zip(batchs, range(len(batchs))):
                if verbose and j % 20 == 0:
                    print('batch {} / {}'.format(i, len(batchs)))
                self.update_mini_batch(batch, learning_rate, mu=next_mu, v=v)
            
            if test_data is not None:
                train_evl = self.evaluate(training_data)
                val_evl = self.evaluate(test_data)
                print('Epoch {}, training accuracy: {} / {} ({}%); validation accuracy: {} / {} ({}%)'.format(i, train_evl, len(training_data), str(100*train_evl/len(training_data))[:5], 
                                                                                                              val_evl, len(test_data), str(100*val_evl/len(test_data))[:5]))
            else:
                print('Epoch {} complete'.format(i))
            if mu is not None:
                next_mu = mu + friction_range * (i/epochs)
            else:
                learning_rate *= decreasing_rate
                #print(learning_rate)
            
    # x: inputs column by column
    # y: targets column by column
    def backpropagation(self, x, y):   
        bias_gradients = {}
        weight_gradients = {}
        
        # 1. feedforward
        #print(x)
        activations = {0:x} # save all 'a' layer by layer
        activation = activations[0]
        zs = {} # save all 'z'(inputs) layer by layer
        for i in range(1, self.num_layers+1):
            w = self.weights[i]
            b = self.biases[i]
            z = np.dot(w, activation) + b
            #print(z)
            zs[i] = z
            if i == self.num_layers: # output layer, use softmax
                activation = self.softmax(z)
                #print(np.sum(activation))
            else:
                activation = self.f(z)
            self.seesee[i] = activation
            activations[i] = activation
            
        # 2. backpropagation
        # 2.1 compute delta of output layer
        deltas = {}
        delta = None                                 
        deltas[self.num_layers-1] = 0
        # 2.2 compute delta of each layer
        for i in range(self.num_layers, 0, -1):
            z = zs[i]
            activation = activations[i]
            # output layer
            if i == self.num_layers:
                #print(np.sum(activation), np.sum(y))
                #delta = (activation - y) * self.f(z,derivative=True)
                delta = activation - y
                deltas[i] = delta
            else:
                da_dz = self.f(z, derivative=True)
                delta = deltas[i+1] # delta l+1
                delta = np.dot(self.weights[i+1].T, delta) * da_dz # delta l
                deltas[i] = delta
        #print(zs)
        #print(deltas)
        # 2.3 compute delta_(l) * a_(l-1).T
        for i in range(self.num_layers, 0, -1):
            weight_gradients[i] = np.dot(deltas[i], activations[i-1].T)
            bias_gradients[i] = deltas[i]
        #print(weight_gradients)
        return weight_gradients, bias_gradients
            
    def update_mini_batch(self, batch_data, learning_rate, mu, v):
        """update by mini batch
        
        Args:
            batch_data (numpy.ndarray): the batch data points
            learning_rate (float): learning rate
            v (float) for momentum-like methods
        Returns:
            None
        """
        weight_adjustments = {}
        bias_adjustments = {}
        for x, y in batch_data:
            # dC/dw and dC/db of each layer
            weight_gradients, bias_gradients = self.backpropagation(x, y)
            
            # one batch update once, sum up all gradients first
            for i in range(1, self.num_layers+1):
                weight_adjustments[i] = weight_adjustments.get(i, np.zeros(shape=weight_gradients[i].shape)) + weight_gradients[i]
                bias_adjustments[i] = bias_adjustments.get(i, np.zeros(shape=bias_gradients[i].shape)) + bias_gradients[i]
        self.adjust = weight_adjustments
        # update with mean of gradients
        for i in range(self.num_layers, 0, -1):
            #print(len(batch_data))
            dC_dw = weight_adjustments[i]/len(batch_data)
            dC_db = bias_adjustments[i]/len(batch_data)
            if mu is None:
                self.weights[i] -= learning_rate*dC_dw
                self.biases[i] -= learning_rate*dC_db
            else: # vanilla momentum
                v[0] = mu*v[0] - learning_rate*dC_dw
                v[1] = mu*v[1] - learning_rate*dC_db
                self.weights[i] += v[0]
                self.biases[i] += v[1]
                
    def evaluate(self, test_data):
        """evaluate model
        
        Args:
            test_data (numpy.ndarray): test data
        Returns:
            the correct predicted number
        """
        #print(type(test_data[0][1]))
        if isinstance(test_data[1], np.ndarray):
            #print(1)
            results = [(np.argmax(self.predict(x)), y) for (x, y) in test_data]
        else:
            #print(2)
            results = [(np.argmax(self.predict(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)
    
    def draw_boundary(self, dataset):
        pass
    


img_width = 30
img_height = 30
file_path = 'Data_Train'
class_names = ['Class1', 'Class2', 'Class3']
int_to_class = {0:'Class1', 1:'Class2', 2:'Class3'}

train_df, train_class, val_df, val_class = get_data(file_path=file_path, class_names=class_names, 
                                                    nb_samples=[1000,1000,1000], data_augmentation=False)

training, validation = preprocessing(train_df, train_class, val_df, val_class, class_names=class_names, pca=True)


#del NN
NN = NeuronNetwork()
NN.add_layer(100, initialization='scaled_normal_relu', input_size=training[0][0].shape[0])
#NN.add_layer(10, initialization='scaled_normal_relu')
#NN.add_layer(50)
NN.fit(training_data=training, epochs=100, activation_function='relu', batch_size=1,
       learning_rate = 0.1, decreasing_rate=0.99, test_data=validation, verbose=False)