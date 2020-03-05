import numpy as np

import keras
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.initializers import *
from keras.regularizers import l1, l2, l1_l2

def LeNet_5(HP, input_shape):
    
    seed = HP["seed"]
    drop1 = HP["drop1"]
    drop2 = HP["drop2"]
    k_init = glorot_uniform(seed)
    k1 = int(HP["k_size_1"])
    k2 = int(HP["k_size_2"])
    print("k1,k2", k1,k2)
    print("drop1, drop2", drop1, drop2)
    
    k_init_dense = glorot_uniform(seed=seed)
    k_init_conv = glorot_uniform(seed=seed) # he_normal(seed=seed)
    
    model = Sequential()
    model.add(Conv2D(filters = 6, 
                 kernel_size = k1, #5, 
                 strides = 1, 
                 activation = 'relu', 
                 input_shape = input_shape, kernel_initializer = k_init_conv))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    model.add(Conv2D(filters = 16, 
                 kernel_size = k2, # 5,
                 strides = 1,
                 activation = 'relu',
                 input_shape = (14,14,6), kernel_initializer = k_init_conv))
    
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    model.add(Dropout(drop1))
    model.add(Flatten())
    model.add(Dense(units = 120, activation = 'relu', kernel_initializer=k_init_dense))
    model.add(Dense(units = 84, activation = 'relu', kernel_initializer=k_init_dense))
    model.add(Dropout(drop2))
    model.add(Dense(units = 10, activation = 'softmax', kernel_initializer=k_init_dense))
    
    return model

def LeNet_new(HP, input_shape):
    
    seed = HP["seed"]
    reg1 = HP["reg1"]
    reg2 = HP["reg2"]
    drop1 = HP["drop1"]
    drop2 = HP["drop2"]
    input_shape = HP["input_shape"]
    num_classes = HP["num_classes"]
    k1 = int(HP["k_size_1"])
    k2 = int(HP["k_size_2"])
    ps = int(HP["p_size"])
    
    reg = l1_l2(reg1, reg2)
    
    k_init_dense = glorot_uniform(seed=seed)
    k_init_conv = he_normal(seed=seed)
    
    print("k1",k1,"k2",k2,"ps",ps)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(k1, k1), activation='relu', input_shape=input_shape, kernel_initializer=k_init))
    model.add(Conv2D(64, (k2, k2), activation='relu', kernel_initializer=k_init))
    model.add(MaxPooling2D(pool_size=(ps, ps)))
    model.add(Dropout(drop1))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer = k_init, kernel_regularizer = reg))
    model.add(Dropout(drop2))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer = k_init, kernel_regularizer = reg))

    return model

def val_split_CNN(X,Y,frac):    

    np.random.seed(10)
    L = X.shape[0]
    ind = np.random.choice(L,int(L*frac),replace = False)
    train_X = X[ind,:,:,:]
    Y = np.array(Y) 

    train_Y = Y[ind,:]
    ind = ind.tolist()
    ind1 = list(set(np.arange(L))-set(ind))

    validation_X = X[ind1,:,:,:]
    validation_Y = Y[ind1,:]

    return train_X, train_Y, validation_X, validation_Y

def model_reset(hparameters):
    
    model_type = hparameters['model_type']
    input_shape = hparameters['input_shape']
    metrics = hparameters['metrics']
    print("model_reset: model_type, input_shape", model_type, input_shape)
    
    # optimizer = hparameters['optimizer']
    optimizer = set_opt(hparameters)
    loss = hparameters['loss']
    
    if model_type == "ResNet50":
        model = ResNet50(hparameters, input_shape= input_shape)
    elif model_type == "LeNet_1":
        model = LeNet_1(hparameters, input_shape= input_shape)
    elif model_type == "LeNet_2":
        model = LeNet_1(hparameters, input_shape= input_shape)
    elif model_type == "LeNet_5":
        model = LeNet_5(hparameters, input_shape= input_shape)            
    elif model_type == "DenseNet":
        model = DenseNet(hparameters, input_shape= input_shape)
    else:
        print("model type not defined!")
        
    model.compile(optimizer = optimizer, loss = loss, metrics = [metrics])
            
    return model
    
def model_main_cnn(X_train, Y_train, X_val, Y_val, X_test, Y_test, hparameters, model_in = None):
    
    print("X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape", X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)
    # optimizer = hparameters['optimizer']
    lr = hparameters['lr']
    momentum = hparameters['momentum']
    beta_1 = hparameters['beta_1']
    beta_2 = hparameters['beta_2']
    decay = hparameters['decay']
    rho = hparameters['rho']
    schedule_decay = hparameters['schedule_decay']
    loss = hparameters['loss']
    metrics = hparameters['metrics']
    epochs = hparameters['epochs']
    batch_size = int(hparameters['batch_size'])
    early_stopping = hparameters['early_stopping']
    classes = hparameters['num_classes']
    inherit = hparameters['inherit']
    input_shape = hparameters['input_shape']
    model_type = hparameters['model_type']
    val_frac = hparameters['val_frac']
    print("lr:", lr)
    
    # optimizer = set_opt(hparameters)
    
    # train_X, train_Y, val_X, val_Y = val_split_CNN(X_train,Y_train,val_frac)
    print("model_main_cnn: model_type, input_shape", model_type, input_shape)
    if model_in!=None and inherit==True:
        model = model_in
    else:
        if model_type == "ResNet50":
            model = ResNet50(hparameters, input_shape= input_shape)
        elif model_type == "LeNet_1":
            model = LeNet_1(hparameters, input_shape= input_shape)
        elif model_type == "LeNet_2":
            model = LeNet_1(hparameters, input_shape= input_shape)
        elif model_type == "LeNet_5":
            model = LeNet_5(hparameters, input_shape= input_shape)            
        elif model_type == "DenseNet":
            model = DenseNet(hparameters, input_shape= input_shape)
        else:
            print("model type not defined!")
    
    optimizer = set_opt(hparameters)
    model.compile(optimizer = optimizer, loss = loss, metrics = [metrics])
    
    history = LossHistory()
    if early_stopping == True:
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    else:
        callback = history
    
    print("Go!!!!!!!")
    h = model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_val, Y_val), verbose = 0, callbacks = [callback])
    eva_val = model.evaluate(X_val, Y_val)
    eva_test = model.evaluate(X_test, Y_test)
    eva_val_history = h.history['val_loss']
    
    print("h.history['val_loss']", h.history['val_loss'])
    # nb_epochs = len(h.history['val_loss'])
    # print("model.nb_epoch:", nb_epochs)
    
    # print("eva:", eva)
    
    return eva_val, eva_test, model, eva_val_history

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def set_opt(hparameters):
    
    opt = hparameters["optimizer"]
    lr = hparameters["lr"]
    rho = hparameters["rho"]
    momentum = hparameters["momentum"]
    beta_1 = hparameters["beta_1"]
    beta_2 = hparameters["beta_2"]
    decay = hparameters["decay"]
    schedule_decay = hparameters["schedule_decay"]
    print("set_opt:lr", lr)

    if opt=="SGD":
        optimizer=keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True)
    if opt=="RMSprop":
        optimizer=keras.optimizers.RMSprop(lr=lr, rho=rho, epsilon=None, decay=decay)
    if opt=="Adagrad":
        optimizer=keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=decay)
    if opt=="Adadelta":
        optimizer=keras.optimizers.Adadelta(lr=lr, rho=rho, epsilon=None, decay=decay)
    if opt=="Adam":
        optimizer=keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=None, decay=decay)
    if opt=="Adamax":
        optimizer=keras.optimizers.Adamax(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=None, decay=decay)
    if opt=="Nadam":
        optimizer=keras.optimizers.Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=None, schedule_decay=schedule_decay)

    return optimizer
