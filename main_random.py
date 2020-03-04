import time
import matplotlib.pyplot as plt
from time import gmtime, strftime

# MNIST and CIFAR-10

# MNIST
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Fashion FMNIST
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

# CIFAR10
# (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

num_classes = 10

L_train = X_train.shape[0]

X_train_0, Y_train_0, X_test_0, Y_test_0 = X_train, Y_train, X_test, Y_test

HPcv_0 = {'drop1':np.linspace(0,0.6,100), 'drop2':np.linspace(0,0.6,100), 
          "lr":np.exp(np.linspace(np.log(0.0001), np.log(0.1), 300))}

HPcv = {"batch_size": np.exp(np.linspace(np.log(20), np.log(200), 100)), 'k_size_1':np.linspace(1,6,100), 'k_size_2':np.linspace(1,6,100), 
          'drop1':np.linspace(0,0.6,100), 'drop2':np.linspace(0,0.6,100),
          "lr":np.exp(np.linspace(np.log(0.0001), np.log(0.1), 300))} 

HP = {"model_code":"DL_CNN", "seed":10, "num_classes": 10, "epochs":28, "batch_size": 100, "val_frac": 0.80,  
      "reg1":0, "reg2":0.02, "loss": "categorical_crossentropy", "lr":0.03, "rho":0.9, "momentum": 0.9, "beta_1":0.9, 
      "beta_2":0.999, "decay":0.0, "schedule_decay":0.004, "model_type":"LeNet_5", "optimizer":"Adam", "metrics":"categorical_accuracy",
        "k_size_1":3, "k_size_2":3, "p_size":2, "drop1":0.25, "drop2":0.5, "early_stopping": False, "patience":3, "inherit": False}

HHP={"n_sample": 20, "iteration":False, "para_iter":False}
val_frac = HP["val_frac"]

for i in range(10):
    
    ind = np.random.choice(L_train,5000,replace = False)
    ind_train = ind.tolist()

    X_train = X_train_0 # [ind_train,:,:]
    Y_train = Y_train_0 # [ind_train]

    L_test = X_test.shape[0]
    ind1 = np.random.choice(L_test,5000,replace = False)
    ind_test = ind1.tolist()

    X_test = X_test_0 # [ind_test,:,:]
    Y_test = Y_test_0 # [ind_test]

    
    img_rows = X_train.shape[1]
    img_cols = X_train.shape[2]

    try:
        channels = X_train.shape[3]
    except:
        channels = 1

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], channels, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)
    
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)
    
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    HP["input_shape"] = input_shape    

    train_X, train_Y, val_X, val_Y = val_split_CNN(X_train, Y_train, val_frac)
    data = train_X, train_Y, val_X, val_Y
    model = model_main_cnn

    eva_val_list, eva_test_list = HP_search_rand(data, HP, HPcv, HHP, d_HP = None, parameters = None)
    
    eva_val_min = np.min(eva_val_list)
    eva_test_min = np.min(eva_test_list)
    
    datetime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    fn = "Results/HyperTube/FMNIST/" + datetime + "hp_seach_random" #+ str(val)

    # ind = np.arange(2)+1
    rows = zip(eva_val_list, eva_test_list)

    with open(fn, "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
        #f.write(str(best_HP))*
