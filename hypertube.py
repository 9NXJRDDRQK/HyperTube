import numpy as np
import random
from random import shuffle
import sys
sys.path.append('../')

# from Utils_Research.DL_FNN.DL_FNN_Model import *
from Models_CNN import *
from Models_RNN import *
from HyperTube_GA import *
from HyperTube_BO import *
from HyperTube_PR import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

import copy
import time
import csv
from time import gmtime, strftime

def loss_history_process(eva_val_history_list, nepoch):
    
    print("loss_history_process:eva_val_history_list", eva_val_history_list)
    min_history = []
    ave_history = []
    
    for j in range(nepoch):

        loss_epoch_j_list = []        
        for i in range(len(eva_val_history_list)):
            loss_epoch_j_list.append(eva_val_history_list[i][j])
        
        loss_epoch_j_min = np.min(loss_epoch_j_list)
        loss_epoch_j_ave = np.mean(loss_epoch_j_list)
        
        min_history.append(loss_epoch_j_min)
        ave_history.append(loss_epoch_j_ave)

    print("loss_history_process: min_history, ave_history", min_history, ave_history)
    return min_history, ave_history


def model_revaluate(model_list, train_X, train_Y, val_X, val_Y, test_X, test_Y, HP_list):

    model_update_list = []
    eva_val_list = []
    eva_test_list = []
    # epochs = HP["epochs"]
    # print("epochs", epochs)
    eva_val_history_list = []
    
    ind = 0
    for model in model_list:
        
        hp = HP_list[ind]
        batch_size = int(hp["batch_size"])
        epochs = hp["epochs"]
        # optimizer = set_opt(hparameters)
        
        history = LossHistory()
        callback = history
        h = model.fit(train_X, train_Y, epochs = epochs, batch_size = batch_size, validation_data = (val_X, val_Y), verbose = 0, callbacks = [callback])
        
        eva_val = model.evaluate(val_X, val_Y)
        eva_test = model.evaluate(test_X, test_Y)
        eva_val_history = h.history['val_loss']
        eva_val_history_list.append(eva_val_history)
        print("h.history['val_loss']", h.history['val_loss'])
        
        model_update_list.append(model)
        eva_val_list.append(eva_val[0])
        eva_test_list.append(eva_test[0])
        ind+=1
    
    return model_update_list, eva_val_list, eva_test_list, HP_list, eva_val_history_list

def model_revaluate_1(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP_list):

    hp_update_list = []
    eva_val_list = []
    eva_test_list = []
    model_update_list = []
    eva_val_history_list = []
    
    for HP in HP_list:
        model_code = HP["model_code"]
        # eva, model = model_apps(model_code, train_X, train_Y, val_X, val_Y, HP, other_utils = None)
        eva_val, eva_test, model, eva_val_history = model_main_cnn(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP)
        # model = model.fit(current_data_X, current_data_Y, validation_data = (val_X, val_Y))
        # eva = model.evaluate(val_X, val_Y)
        
        model_update_list.append(model)
        eva_val_list.append(eva_val[0])
        eva_test_list.append(eva_test[0])
        eva_val_history_list.append(eva_val_history)
    
    return model_update_list, eva_val_list, eva_test_list, HP_list, eva_val_history_list

def model_sample_init(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP, HHP, HPcv, other_utils):
    
    n_init = HHP["n_init"]
    model_code = HP["model_code"]
    HP0 = HP
    
    model_list = []
    eva_val_list = []
    eva_test_list = []
    hp_list = []
    eva_val_history_list = []
    
    HP_random = {}
    
    # print("HPcv.keys():", HPcv.keys())
    # print("len(HPcv.keys()):", len(HPcv.keys()))
    hp_i = 0
    hp_n = len(HPcv.keys())
    
    for hp in HPcv.keys():
        
        if hp_i < (hp_n-1):
            # print("hp_i", hp_i)
            r = HPcv[str(hp)]
            # print("n_init", n_init, "r", len(set(r)))
        
            # HP_random[str(hp)] = random.sample(population = set(r), k = n_init)
            HP_random[str(hp)] = random.choices(r, k = n_init)
            
        hp_i += 1
      
    for i in range(n_init):
        
        HP = copy.deepcopy(HP0)        
        para = None
        print_dict = {}

        for hp in HPcv.keys():
            # print("hp", hp)
            if hp != "time":
                hp_rand = HP_random[str(hp)]
                HP[str(hp)] = hp_rand[i]
        
        if model_code == "CNN":
            eva_val, eva_test, model, eva_val_history = model_main_cnn(train_X, train_Y, test_X, test_Y, val_X, val_Y, HP)
        if model_code == "RNN":
            eva, model = model_main_rnn(train_X, train_Y, val_X, val_Y, test_X, test_Y, hparameters = HP)
        
        # eva, model = model_apps(model_code, train_X, train_Y, val_X, val_Y, HP, other_utils = other_utils)
        try:
            val_cost = eva_val[0]
            test_cost = eva_test[0]
        except:
            val_cost = eva_val
            test_cost = eva_test

        model_list.append(model)
        eva_val_list.append(val_cost)
        eva_test_list.append(test_cost)
        eva_val_history_list.append(eva_val_history)
        hp_list.append(HP)       
    
    return model_list, eva_val_list, eva_test_list, hp_list, eva_val_history_list

def select(model_list, eva_list, eva_val_history_list, hp_list, n_t, adjusted = False):
    
    # print("eva", eva_list)
    # print("n_t", n_t)

    
    eva_list_adjusted = copy.deepcopy(eva_list)
    
    if adjusted == True:
        diff_val_history_list = [item[0]-item[-1] for item in eva_val_history_list]
        print("diff_val_history_list", diff_val_history_list)
        eva_list_adjusted = [(eva_list[i]-diff_val_history_list[i]/2) for i in range(len(diff_val_history_list))]
    
    ind = sorted(range(len(eva_list)), key=lambda i: eva_list_adjusted[i])[:n_t]
    
    # print("ind", ind)
    hp_list = [hp_list[i] for i in ind]
    model_list = [model_list[i] for i in ind]
    eva_list = [eva_list[i] for i in ind]
    # print("eva_list_selected", eva_list)

    return model_list, eva_list, hp_list

def select_1(eva_list, eva_val_history_list, hp_list, n_t, adjusted = False):
    
    # print("eva", eva_list)
    eva_list_adjusted = copy.deepcopy(eva_list)
    
    if adjusted == True:
        diff_val_history_list = [item[0]-item[-1] for item in eva_val_history_list]
        print("diff_val_history_list", diff_val_history_list)
        eva_list_adjusted = [(eva_list[i]-diff_val_history_list[i]/2) for i in range(len(diff_val_history_list))]
    
    ind = sorted(range(len(eva_list)), key=lambda i: eva_list_adjusted[i])[:n_t]
    
    # print("ind", ind)
    hp_list = [hp_list[i] for i in ind]
    eva_list = [eva_list[i] for i in ind]
    # print("eva_list_selected", eva_list)

    return eva_list, hp_list

# target_model.set_weights(model.get_weights()) 

def model_update_sample_1(test_X, test_Y, model_list, HP_list, eva_list, eva_val_history_list, HHP, HPcv, n_t, update_rate, method = None, other_data = None):
    # model_list, hp_list, eva_list, HPcv, HPga, n_t, update_rate, method = update_method
    print("model_update_sample:len(eva_val_history_list)", len(eva_val_history_list))
    print("model_update_sample:eva_val_history_list", eva_val_history_list)
    # prob_list = roulette(eva_list)
    eva_list_adjusted = copy.deepcopy(eva_list)
    
    if adjusted == True:
        diff_val_history_list = [item[0]-item[-1] for item in eva_val_history_list]
        print("diff_val_history_list", diff_val_history_list)
        eva_list_adjusted = [(eva_list[i]-diff_val_history_list[i]/3) for i in range(len(diff_val_history_list))]
    # eva_list_adjusted = eva_list
    
    prob_cross = HHP["prob_cross"]
    n_t_o = int (n_t * (1 - update_rate))
    ind = sorted(range(len(eva_list_adjusted)), key=lambda i: eva_list_adjusted[i])[:n_t_o]
    print("model_update_sample:eva_list_adjusted", eva_list_adjusted)
    # print("model_update_sample: model_list, HP_list", len(model_list), len(HP_list))
    # print("ind_0", ind)
    # HP_list_1 = copy.deepcopy(HP_list[ind])
    HP_list_1 = [HP_list[i] for i in ind]
    # model_list_1 = copy.deepcopy(model_list[ind])
    model_list_1 = [model_list[i] for i in ind]
    eva_list_1 =  [eva_list[i] for i in ind]
    ind_1 = []
    hpcv_list = []
    inherit_list = []
    
    history = other_data
    
    """
    for i in ind:
        hp_i = HP_list_1[i]
        for hp in HPcv.keys():
            hp_add[str(hp)] = hp_i[str(hp)]
            
        hpcv_list.add(hp_add)
    """

    if method == "GA" and n_t_o>0:
        HPga = HHP["HPga"]
        prob = HHP["mutation_prob"]
        cross_mutation_prob = HHP["cross_mutation_prob"]
        # crossover
        for i in range(int((n_t - n_t_o)/2)+1):
            # print("ind", len(ind))
            # print("(int(n_t_o/2)+1)", int(n_t_o/2)+1)
            
            print("crossover loop next iteration")
            
            """
            individual_inds = random.sample(ind[:int(n_t_o/3)+1], k=2)
            # individual_inds = random.choice(ind[:int(n_t_o/3)+1], size=2, replace=False)
            hp_individual_1, hp_individual_2 = HP_list[individual_inds[0]], HP_list[individual_inds[1]]
            hp_individual_1_0 = copy.deepcopy(hp_individual_1)
            hp_individual_1, hp_individual_2 = crossover_hp(HPga, hp_individual_1, hp_individual_2, prob_cross)
            print("new_individual", hp_individual_1)
            print("hp_individual_0", hp_individual_1_0)
                
            if hp_individual_1 == hp_individual_1_0:
                print("same")
                continue
            else:
                print("cross!")
                HP_list_1.append(hp_individual_1)
                HP_list_1.append(hp_individual_2)  
            
            """
            try:
                # individual_inds = random.sample(ind[:np.max([int(n_t_o/4)+1,2])], k=2)
                # individual_inds = random.choice(ind[:int(n_t_o/3)+1], size=2, replace=False)
                individual_inds = random.sample(np.arange(np.max([int(n_t/5),2])), k=2) 
                
                hp_individual_1, hp_individual_2 = copy.deepcopy(HP_list_1[individual_inds[0]]), copy.deepcopy(HP_list_1[individual_inds[1]])
                hp_individual_1_0 = copy.deepcopy(hp_individual_1)
                hp_individual_2_0 = copy.deepcopy(hp_individual_2)
                hp_individual_1, hp_individual_2 = crossover_hp(HPga, hp_individual_1, hp_individual_2, prob_cross)
                new_individual_1 = mutation_hp(hp_individual_1, cross_mutation_prob, HPga)
                new_individual_2 = mutation_hp(hp_individual_2, cross_mutation_prob, HPga)

                if new_individual_1 == hp_individual_1_0 or new_individual_1 == hp_individual_2_0:
                    print("same")
                    continue
                else:
                    print("cross!")
                    print("crossover:len(HP_list_1):", len(HP_list_1))
                    for hp in HPcv.keys():
                        print("new_individual_1:", str(hp), new_individual_1[str(hp)])
                    print("crossover:len(HP_list_1):", len(HP_list_1)+1)
                    for hp in HPcv.keys():
                        print("new_individual_2:", str(hp), new_individual_2[str(hp)])
                        
                    HP_list_1.append(new_individual_1)
                    HP_list_1.append(new_individual_2)
                    inherit_list.append(np.min(individual_inds))
                    inherit_list.append(np.min(individual_inds))
                    print("individual_inds", individual_inds)
                    print("inherit_list", inherit_list)        
            except:
                print("no sufficient individuals from cross-over")
            
            print("len(HP_list_1)", len(HP_list_1))
            """
            individual_inds = random.sample(ind[:(int(n_t_o/2)+1)], k=2)
            print("individual_inds", individual_inds)
            hp_individual_1, hp_individual_2 = HP_list[individual_inds[0]], HP_list[individual_inds[1]]
            hp_individual_1_0 = copy.deepcopy(hp_individual_1)
            hp_individual_1, hp_individual_2 = crossover_hp(HPga, hp_individual_1, hp_individual_2, prob_cross)
            if hp_individual_1 == hp_individual_1_0:
                print("same")
                continue
            else:
                print("cross!")
                HP_list_1.append(hp_individual_1)
                HP_list_1.append(hp_individual_2)  
            """
        if len(HP_list_1) > n_t:
            HP_list_1 = HP_list_1[:n_t]
        # mutation
        # for i in range(n_t - n_t_o):  
        while len(HP_list_1) < n_t:
            # hp_individual = HP_list[random.choice(ind)]
            mutation_ind = random.choice(np.arange(int(n_t/5)+1))
            # hp_individual = copy.deepcopy(HP_list[random.choice(ind[:(int(n_t_o/4)+1)])])
            hp_individual = copy.deepcopy(HP_list_1[mutation_ind])
            hp_individual_0 = copy.deepcopy(hp_individual)
            new_individual = mutation_hp(hp_individual, prob, HPga)
            print("mutation:len(HP_list_1):", len(HP_list_1))
            for hp in HPcv.keys():
                print("new_individual:", str(hp), new_individual[str(hp)])
                # print("hp_individual_0:", str(hp), hp_individual_0[str(hp)])
            if new_individual!=hp_individual_0:
                print("mutation!")
                HP_list_1.append(new_individual)
                print("mutation_ind_0", mutation_ind)
                mutation_ind = random.choice([0, mutation_ind])
                inherit_list.append(mutation_ind)
                print("mutation_ind", mutation_ind)
                print("inherit_list")
            else:
                print("same")
            # if len(model_list_1) == n_t:
            #    break
            # print("HP_list_1", HP_list_1)
            # target_model.set_weights(model.get_weights()) 
        for i in range(n_t - n_t_o):
            
            print("n_t_o + i:", n_t_o + i)
            for hp in HPcv.keys():
                new_individual = HP_list_1[n_t_o + i]
                print("new_individual:", str(hp), new_individual[str(hp)])
            new_model = model_reset(HP_list_1[n_t_o + i])
            # print("ind:", ind)
            # ind_copy = random.choice(ind[:np.max([int(n_t_o/4),1])])
            # print("ind_copy:", ind_copy)
            # ind_copy = random.choice(np.arange(np.max([int(n_t_o/4),1])))
            # print("ind_copy", ind_copy)
            print("len(model_list_1)", len(model_list_1))
            # new_model.set_weights(model_list_1[ind_copy].get_weights())
            print("inherit_list[i]", inherit_list[i])
            new_model.set_weights(model_list_1[inherit_list[i]].get_weights())
            model_list_1.append(new_model)
        HP_list_new = HP_list_1
        model_list_new = model_list_1     
    elif method == "BO" and n_t_o>0:   
        # acq = expected_improvement
        history = other_data
        HP_list_new, model_list_new, history = update_BO(test_X, test_Y, model_list, HP_list, eva_list, HHP, HPcv, n_t, other_data = history)
        HP_list_new = HP_list_1
        model_list_new = model_list_1
    
    elif method == "predict" and n_t_o>0:
        history = other_data
        HP_list_new, model_list_new, history = update_predict(test_X, test_Y, model_list, HP_list, eva_list, HHP, HPcv, n_t, other_data = history)
        HP_list_new = HP_list_1
        model_list_new = model_list_1
    else:
        HP_list_new = HP_list
        model_list_new = model_list
        
    # if method == "Predict":
            
    return HP_list_new, model_list_new, history

def model_online(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP, HHP, HPcv, fn, other_utils):
    
    model_code = HP["model_code"]
    level = HHP["level"]
    n_init = HHP["n_init"]
    setting = HHP["setting"]
    n_sample = train_X.shape[0]
    n_level = len(level)
    n_data0 = 0
    eva_val_min_list = []
    eva_test_min_list = []
    meva_val_list = []
    meva_test_list = []
    hpe0 = HP["epochs"]
    t = 0
    update_method = HHP["update_method"]
    update_rate = HHP["update_rate"]
    n_t_list = []
    e_t_list = []
    d_t_list = []
    d0_t_list = []
    min_history_full = []
    ave_history_full = []
    
    history_hp = None
    
    # print("len(train_X.shape), len(train_Y.shape)", len(train_X.shape), len(train_Y.shape))
    
    # for i in range(len(hp_list)):            
    #    hp_list[i]["time"] = 0
       
    if t==0:
        n_data = int(n_sample * level[0])
        # print("n_data", n_data, "yndata", type(n_data))
        HP["epochs"] = int(hpe0 * level[0])
        ep = HP["epochs"]
        
        if len(train_X.shape)==4 and len(train_Y.shape)==2:
            train_mX, train_mY = train_X[:n_data,:,:,:], train_Y[:n_data,:]
        if len(train_X.shape)==3 and len(train_Y.shape)==3:
            train_mX, train_mY = train_X[:n_data,:,:], train_Y[:n_data,:,:]
        if len(train_X.shape)==3 and len(train_Y.shape)==2:
            train_mX, train_mY = train_X[:n_data,:,:], train_Y[:n_data,:]
        if len(train_X.shape)==2 and len(train_Y.shape)==1: 
            train_mX, train_mY = train_X[:n_data,:], train_Y[:n_data]
        """
        else:
            if len(train_X.shape)==4 and train_Y==None:
                train_mX, train_mY = train_X[:n_data,:,:,:], None
        """
        
        HP["m"] = train_mX.shape[0]
        
        model_list, eva_val_list, eva_test_list, hp_list, eva_val_history_list = model_sample_init(train_mX, train_mY, val_X, val_Y, test_X, test_Y, HP, HHP, HPcv, other_utils)
        t = t + 1
        eva_val_min_list.append(min(eva_val_list))
        meva_val_list.append(np.mean(eva_val_list))
        
        eva_test_min_list.append(min(eva_test_list))
        meva_test_list.append(np.mean(eva_test_list))
        
        min_history, ave_history = loss_history_process(eva_val_history_list, nepoch = ep)
        min_history_full = min_history_full + min_history
        ave_history_full = ave_history_full + ave_history
        print("len(min_history_full)", len(min_history_full))     
        
        n_t_list.append(n_init)
        e_t_list.append(HP["epochs"])
        d_t_list.append(n_data)
        d0_t_list.append(n_data0)
        
        for i in range(len(hp_list)):            
            hp_list[i]["time"] = 1
    
    while t > 0:
        
        """
        x= np.arange(train_X.shape[0])
        x1 = shuffle(x)
        train_X = train_X[x1,:]
        train_Y = train_Y[x1,:]
        """
        
        # eva, model = model_revaluate(model, train_X, train_Y, val_X, val_Y, HP)
        # n_data is for data sample, n_t is for the number of models
        
        if setting == "E" or setting == "P" or setting == "P1":
            n_data0 = n_data
        if setting == "D" or setting == "ED" or setting == "EDR" or setting == "ED_1" or setting == "EDR_1":
            n_data0 = 0
        
        n_data = int(n_sample * level[t])
        
        if len(train_X.shape)==4 and len(train_Y.shape)==2:
            train_mX, train_mY = train_X[n_data0:n_data,:,:,:], train_Y[n_data0:n_data,:]
        if len(train_X.shape)==3 and len(train_Y.shape)==3:
            train_mX, train_mY = train_X[n_data0:n_data,:,:], train_Y[n_data0:n_data,:,:]
        if len(train_X.shape)==3 and len(train_Y.shape)==2:
            train_mX, train_mY = train_X[n_data0:n_data,:,:], train_Y[n_data0:n_data,:]
        if len(train_X.shape)==2 and len(train_Y.shape)==1:
            train_mX, train_mY = train_X[n_data0:n_data,:], train_Y[n_data0:n_data]

        """
        else:        
            if len(train_X.shape)==4 and train_Y==None:
                train_mX, train_mY = train_X[:n_data,:,:,:], None
        """ 
         
        HP["m"] = train_mX.shape[0]
        # train_mX, train_mY = train_X[n_data0:n_data,:,:,:], train_Y[n_data0:n_data,:]
        
        if setting == "E" or setting == "ED" or setting == "EDR":            
            HP["epochs"] = int(hpe0 * level[t])            
            for i in range(len(hp_list)):
                hp_list[i]["epochs"] = int(hpe0 * level[t])
            
        if setting == "ED_1" or setting == "EDR_1":
            HP["epochs"] = int(hpe0 * level[0] * (t+1) **(1/2))
            for i in range(len(hp_list)):
                hp_list[i]["epochs"] = int(hpe0 * level[0] * (t+1) **(1/2))
            # print("HP_epochs", HP["epochs"])
            
        # if setting == "P1":
        #    HP["epochs"] = int(hpe0 * level[t])
            
        print("setting:", setting)
        ep = HP["epochs"]
            
        if setting == "D" or setting == "E":
            n_t = int((n_sample / (n_data * n_level))  * n_init)
                     
        if setting == "ED" or setting == "EDR":
            n_t = int(((n_sample / (n_data * n_level))**2) * n_init)
        if setting == "ED_1" or setting == "EDR_1":
            n_t = int(((n_sample / (n_data * n_level))**(3/2)) * n_init)
        if setting == "P":
            n_t = n_init
            
        print("n_sample, n_data, n_level, n_init, n_t:", n_sample, n_data, n_level, n_init, n_t)          
        # print("HP_epochs, n_t", HP["epochs"], n_t)
        n_t_list.append(n_t)
        e_t_list.append(HP["epochs"])
        d_t_list.append(n_data)
        d0_t_list.append(n_data0)
        
        if update_method != None and t<int(n_level/2):           
            hp_list, model_list, history_hp = model_update_sample(val_X, val_Y, model_list, hp_list, eva_val_list, HHP, HPcv, n_t, update_rate, method = update_method, other_data = history_hp)
        elif setting == "E" or setting == "D" or setting == "P" or setting == "EDR" or setting == "EDR_1":
            model_list, eva_list, hp_list = select(model_list, eva_val_list, hp_list, n_t)
        elif setting == "ED" or setting == "ED_1":
            eva_list, hp_list = select_1(eva_val_list, hp_list, n_t)
        else:
            print("no such setting!")
        
        
        for i in range(len(hp_list)):            
            hp_list[i]["time"] = t+1
        
        # print(eva_list)
        
        if setting == "E" or setting == "D" or setting == "P" or setting == "EDR" or setting == "EDR_1":
            model_list, eva_val_list, eva_test_list, hp_list, eva_val_history_list = model_revaluate(model_list, train_mX, train_mY, val_X, val_Y, test_X, test_Y, hp_list)
        if setting == "ED" or setting == "ED_1":
            model_list, eva_val_list, eva_test_list, hp_list, eva_val_history_list = model_revaluate_1(train_mX, train_mY, val_X, val_Y, test_X, test_Y, HP_list = hp_list)
        # model_update_list, eva_val_list, eva_test_list, HP_list
        min_history, ave_history = loss_history_process(eva_val_history_list, nepoch = ep)
        
        # print("len(min_history_full), len(min_history)", len(min_history_full), len(min_history))
        # print("min_history_full, min_history", min_history_full, min_history)    
        min_history_full = min_history_full + min_history
        ave_history_full = ave_history_full + ave_history
        # print("min_history_full", min_history_full)           
        # model_list, eva_list, hp_list = model_sample_init(train_mX, train_mY, val_X, val_Y, HP, HHP, HPcv)
        t = t + 1
        print("eva_list", eva_val_list)
        eva_val_min_list.append(min(eva_val_list))
        eva_test_min_list.append(min(eva_test_list))
        meva_val_list.append(np.mean(eva_val_list))
        meva_test_list.append(np.mean(eva_test_list))
        if t == n_level:
            break
    
    print("min_history_full, ave_history_full", min_history_full, ave_history_full)
    datetime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    rows = zip(eva_val_min_list, eva_test_min_list, meva_val_list, meva_test_list, n_t_list, e_t_list, d_t_list, d0_t_list)
    with open(datetime + fn, "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    
    fn = "_full_"+fn
    datetime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    rows_1 = zip(min_history_full, ave_history_full)
    with open(datetime + fn, "w") as f:
        writer = csv.writer(f)
        for row in rows_1:
            writer.writerow(row)
    
    try:
        index_min = np.argmin(eva_val_min_list)
        hp_list_min = hp_list[index_min]
        hp_list_min["epochs"] = 25
        # eva_min, model = model_apps(model_code, train_X, train_Y, val_X, val_Y, HP = hp_list_min, other_utils = other_utils)
        eva_val_final, eva_test_final, model, eva_val_history = model_main_cnn(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP = hp_list_min)
    
        fn_1 = fn + "_final_test"
        with open(datetime + fn_1, "w") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
    except:
        print("test hp fail!")
    
    return model_list, eva_val_min_list, eva_test_min_list, meva_val_list, meva_test_list, min_history_full, ave_history_full
    
    # return model_list, eva_val_min_list, eva_test_min_list, meva_val_list, meva_test_list, hp_list_min, eva_val_final, eva_test_final 
    
def model_sample_init_hp(HP, HHP, HPcv):
    
    n_init = HHP["n_init"]
    model_code = HP["model_code"]
    HP0 = HP
    
    model_list = []
    eva_val_list = []
    eva_test_list = []
    hp_list = []
    eva_val_history_list = []
    
    HP_random = {}
    hp_i = 0
    hp_n = len(HPcv.keys())
    
    for hp in HPcv.keys():
        
        if hp_i < (hp_n-1):
            r = HPcv[str(hp)]
            HP_random[str(hp)] = random.choices(r, k = n_init)
            
        hp_i += 1
        
    return HP_random
    
def model_sample_init_1(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP, HHP, HP_random, other_utils):
    
    n_init = HHP["n_init"]
    model_code = HP["model_code"]
    HP0 = HP
    
    model_list = []
    eva_val_list = []
    eva_test_list = []
    hp_list = []
    eva_val_history_list = []
          
    for i in range(n_init):
        
        HP = copy.deepcopy(HP0)        
        para = None
        print_dict = {}

        for hp in HP_random.keys():
            # print("hp", hp)
            if hp != "time":
                hp_rand = HP_random[str(hp)]
                HP[str(hp)] = hp_rand[i]
        
        if model_code == "CNN":
            eva_val, eva_test, model, eva_val_history = model_main_cnn(train_X, train_Y, test_X, test_Y, val_X, val_Y, HP)
        if model_code == "RNN":
            eva, model = model_main_rnn(train_X, train_Y, val_X, val_Y, test_X, test_Y, hparameters = HP)
        
        # eva, model = model_apps(model_code, train_X, train_Y, val_X, val_Y, HP, other_utils = other_utils)
        try:
            val_cost = eva_val[0]
            test_cost = eva_test[0]
        except:
            val_cost = eva_val
            test_cost = eva_test

        model_list.append(model)
        eva_val_list.append(val_cost)
        eva_test_list.append(test_cost)
        eva_val_history_list.append(eva_val_history)
        hp_list.append(HP)       
    
    return model_list, eva_val_list, eva_test_list, hp_list, eva_val_history_list

def model_update_sample(test_X, test_Y, model_list, HP_list, eva_list, eva_val_history_list, HHP, HPcv, n_t, update_rate, method = None, other_data = None, adjusted = False):
    # model_list, hp_list, eva_list, HPcv, HPga, n_t, update_rate, method = update_method
    print("model_update_sample:len(eva_val_history_list)", len(eva_val_history_list))
    print("model_update_sample:eva_val_history_list", eva_val_history_list)
    # prob_list = roulette(eva_list)
 
    #diff_val_history_list = [item[0]-item[-1] for item in eva_val_history_list]
    #print("diff_val_history_list", diff_val_history_list)
    #eva_list_adjusted = [(eva_list[i]-diff_val_history_list[i]/3) for i in range(len(diff_val_history_list))]
    # eva_list_adjusted = eva_list
    
    eva_list_adjusted = copy.deepcopy(eva_list)
    
    if adjusted == True:
        diff_val_history_list = [item[0]-item[-1] for item in eva_val_history_list]
        print("diff_val_history_list", diff_val_history_list)
        eva_list_adjusted = [(eva_list[i]-diff_val_history_list[i]/3) for i in range(len(diff_val_history_list))]
    
    prob_cross = HHP["prob_cross"]
    n_t_o = int (n_t * (1 - update_rate))
    ind = sorted(range(len(eva_list_adjusted)), key=lambda i: eva_list_adjusted[i])[:n_t_o]
    print("model_update_sample:eva_list_adjusted", eva_list_adjusted)
    print("model_update_sample:eva_list_adjusted (ordered)", [eva_list_adjusted[i] for i in ind])
    if len(ind)>1:
        prob_list = roulette([eva_list_adjusted[i] for i in ind])
    else:
        prob_list = [1]
    print("prob_list:", prob_list)
    # print("model_update_sample: model_list, HP_list", len(model_list), len(HP_list))
    # print("ind_0", ind)
    # HP_list_1 = copy.deepcopy(HP_list[ind])
    HP_list_1 = [HP_list[i] for i in ind]
    # model_list_1 = copy.deepcopy(model_list[ind])
    model_list_1 = [model_list[i] for i in ind]
    eva_list_1 =  [eva_list[i] for i in ind]
    ind_1 = []
    hpcv_list = []
    inherit_list = []
    
    history = other_data
    
    """
    for i in ind:
        hp_i = HP_list_1[i]
        for hp in HPcv.keys():
            hp_add[str(hp)] = hp_i[str(hp)]
            
        hpcv_list.add(hp_add)
    """
    
    print("n_t_o", n_t_o)

    if method == "GA" and n_t_o>0:
        HPga = HHP["HPga"]
        prob = HHP["mutation_prob"]
        cross_mutation_prob = HHP["cross_mutation_prob"]
        # crossover
        for i in range(int((n_t - n_t_o)/2)+1):
            # print("ind", len(ind))
            # print("(int(n_t_o/2)+1)", int(n_t_o/2)+1)
            
            print("crossover loop next iteration")
            
            try:
                # individual_inds = random.sample(ind[:np.max([int(n_t_o/4)+1,2])], k=2)
                # individual_inds = random.choice(ind[:int(n_t_o/3)+1], size=2, replace=False)
                # individual_inds = random.sample(np.arange(np.max([int(n_t/5),2])), k=2) 
                individual_inds = np.random.choice(np.arange(n_t_o), 2, p=prob_list, replace=False)
                print("individual_inds", individual_inds)
                hp_individual_1, hp_individual_2 = copy.deepcopy(HP_list_1[individual_inds[0]]), copy.deepcopy(HP_list_1[individual_inds[1]])
                hp_individual_1_0 = copy.deepcopy(hp_individual_1)
                hp_individual_2_0 = copy.deepcopy(hp_individual_2)
                hp_individual_1, hp_individual_2 = crossover_hp(HPga, hp_individual_1, hp_individual_2, prob_cross)
                new_individual_1 = mutation_hp(hp_individual_1, cross_mutation_prob, HPga)
                new_individual_2 = mutation_hp(hp_individual_2, cross_mutation_prob, HPga)

                if new_individual_1 == hp_individual_1_0 or new_individual_1 == hp_individual_2_0:
                    print("same")
                    continue
                else:
                    print("cross!")
                    print("crossover:len(HP_list_1):", len(HP_list_1))
                    for hp in HPcv.keys():
                        print("new_individual_1:", str(hp), new_individual_1[str(hp)])
                    print("crossover:len(HP_list_1):", len(HP_list_1)+1)
                    for hp in HPcv.keys():
                        print("new_individual_2:", str(hp), new_individual_2[str(hp)])
                        
                    HP_list_1.append(new_individual_1)
                    HP_list_1.append(new_individual_2)
                    inherit_list.append(np.min(individual_inds))
                    inherit_list.append(np.min(individual_inds))
                    print("individual_inds", individual_inds)
                    print("inherit_list", inherit_list)        
            except:
                print("no sufficient individuals from cross-over")
            
            print("len(HP_list_1)", len(HP_list_1))

        if len(HP_list_1) > n_t:
            HP_list_1 = HP_list_1[:n_t]
        # mutation
        # for i in range(n_t - n_t_o):  
        while len(HP_list_1) < n_t:
            # hp_individual = HP_list[random.choice(ind)]
            # mutation_ind = random.choice(np.arange(int(n_t/5)+1))
            mutation_ind = np.random.choice(np.arange(n_t_o), 1, p=prob_list)
            print("mutation_ind", mutation_ind)
            # hp_individual = copy.deepcopy(HP_list[random.choice(ind[:(int(n_t_o/4)+1)])])
            hp_individual = copy.deepcopy(HP_list_1[mutation_ind[0]])
            hp_individual_0 = copy.deepcopy(hp_individual)
            new_individual = mutation_hp(hp_individual, prob, HPga)
            print("mutation:len(HP_list_1):", len(HP_list_1))
            for hp in HPcv.keys():
                print("new_individual:", str(hp), new_individual[str(hp)])
                # print("hp_individual_0:", str(hp), hp_individual_0[str(hp)])
            if new_individual!=hp_individual_0:
                print("mutation!")
                HP_list_1.append(new_individual)
                print("mutation_ind_0", mutation_ind)
                mutation_ind = random.choice([0, mutation_ind])
                inherit_list.append(mutation_ind)
                print("mutation_ind", mutation_ind)
                print("inherit_list")
            else:
                print("same")
            # if len(model_list_1) == n_t:
            #    break
            # print("HP_list_1", HP_list_1)
            # target_model.set_weights(model.get_weights()) 
            
        inherit_list = np.random.choice(np.arange(n_t_o), np.max([(n_t - n_t_o),1]), p=prob_list)
        print("inherit_list_roulette:", inherit_list)
        
        for i in range(n_t - n_t_o):
            
            print("n_t_o + i:", n_t_o + i)
            for hp in HPcv.keys():
                new_individual = HP_list_1[n_t_o + i]
                print("new_individual:", str(hp), new_individual[str(hp)])
            new_model = model_reset(HP_list_1[n_t_o + i])
            # print("ind:", ind)
            # ind_copy = random.choice(ind[:np.max([int(n_t_o/4),1])])
            # print("ind_copy:", ind_copy)
            # ind_copy = random.choice(np.arange(np.max([int(n_t_o/4),1])))
            # print("ind_copy", ind_copy)
            print("len(model_list_1)", len(model_list_1))
            # new_model.set_weights(model_list_1[ind_copy].get_weights())
            print("inherit_list[i]", inherit_list[i])
            new_model.set_weights(model_list_1[inherit_list[i]].get_weights())
            model_list_1.append(new_model)
        HP_list_new = HP_list_1
        model_list_new = model_list_1     
    elif method == "BO" and n_t_o>0:   
        # acq = expected_improvement
        history = other_data
        HP_list_new, model_list_new, history = update_BO(test_X, test_Y, model_list, HP_list, eva_list, HHP, HPcv, n_t, other_data = history)
        HP_list_new = HP_list_1
        model_list_new = model_list_1
    
    elif method == "predict" and n_t_o>0:
        history = other_data
        HP_list_new, model_list_new, history = update_predict(test_X, test_Y, model_list, HP_list, eva_list, HHP, HPcv, n_t, other_data = history)
        HP_list_new = HP_list_1
        model_list_new = model_list_1
    else:
        HP_list_new = HP_list
        model_list_new = model_list
        
    # if method == "Predict":
            
    return HP_list_new, model_list_new, history

def model_online_1(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP, HHP, HPcv, HP_random, fn, other_utils):
    
    model_code = HP["model_code"]
    level = HHP["level"]
    n_init = HHP["n_init"]
    setting = HHP["setting"]
    n_sample = train_X.shape[0]
    n_level = len(level)
    n_data0 = 0
    eva_val_min_list = []
    eva_test_min_list = []
    meva_val_list = []
    meva_test_list = []
    hpe0 = HP["epochs"]
    t = 0
    update_method = HHP["update_method"]
    update_rate = HHP["update_rate"]
    n_t_list = []
    e_t_list = []
    d_t_list = []
    d0_t_list = []
    min_history_full = []
    ave_history_full = []
    history_hp = None
       
    if t==0:
        n_data = int(n_sample * level[0])
        # print("n_data", n_data, "yndata", type(n_data))
        HP["epochs"] = int(hpe0 * level[0])
        ep = HP["epochs"]
        
        if len(train_X.shape)==4 and len(train_Y.shape)==2:
            train_mX, train_mY = train_X[:n_data,:,:,:], train_Y[:n_data,:]
        if len(train_X.shape)==3 and len(train_Y.shape)==3:
            train_mX, train_mY = train_X[:n_data,:,:], train_Y[:n_data,:,:]
        if len(train_X.shape)==3 and len(train_Y.shape)==2:
            train_mX, train_mY = train_X[:n_data,:,:], train_Y[:n_data,:]
        if len(train_X.shape)==2 and len(train_Y.shape)==1: 
            train_mX, train_mY = train_X[:n_data,:], train_Y[:n_data]
        
        HP["m"] = train_mX.shape[0]
        
        model_list, eva_val_list, eva_test_list, hp_list, eva_val_history_list = model_sample_init_1(train_mX, train_mY, val_X, val_Y, test_X, test_Y, HP, HHP, HP_random, other_utils)
        t = t + 1
        eva_val_min_list.append(min(eva_val_list))
        meva_val_list.append(np.mean(eva_val_list))
        
        eva_test_min_list.append(min(eva_test_list))
        meva_test_list.append(np.mean(eva_test_list))
        
        min_history, ave_history = loss_history_process(eva_val_history_list, nepoch = ep)
        min_history_full = min_history_full + min_history
        ave_history_full = ave_history_full + ave_history
        print("len(min_history_full)", len(min_history_full))     
        
        n_t_list.append(n_init)
        e_t_list.append(HP["epochs"])
        d_t_list.append(n_data)
        d0_t_list.append(n_data0)
        
        for i in range(len(hp_list)):            
            hp_list[i]["time"] = 1
    
    while t > 0:
        
        """
        x= np.arange(train_X.shape[0])
        x1 = shuffle(x)
        train_X = train_X[x1,:]
        train_Y = train_Y[x1,:]
        """
        
        # eva, model = model_revaluate(model, train_X, train_Y, val_X, val_Y, HP)
        # n_data is for data sample, n_t is for the number of models
        
        if setting == "E" or setting == "P" or setting == "P1":
            n_data0 = n_data
        if setting == "D" or setting == "ED" or setting == "EDR" or setting == "ED_1" or setting == "EDR_1":
            n_data0 = 0
        
        n_data = int(n_sample * level[t])
        
        if len(train_X.shape)==4 and len(train_Y.shape)==2:
            train_mX, train_mY = train_X[n_data0:n_data,:,:,:], train_Y[n_data0:n_data,:]
        if len(train_X.shape)==3 and len(train_Y.shape)==3:
            train_mX, train_mY = train_X[n_data0:n_data,:,:], train_Y[n_data0:n_data,:,:]
        if len(train_X.shape)==3 and len(train_Y.shape)==2:
            train_mX, train_mY = train_X[n_data0:n_data,:,:], train_Y[n_data0:n_data,:]
        if len(train_X.shape)==2 and len(train_Y.shape)==1:
            train_mX, train_mY = train_X[n_data0:n_data,:], train_Y[n_data0:n_data]

        """
        else:        
            if len(train_X.shape)==4 and train_Y==None:
                train_mX, train_mY = train_X[:n_data,:,:,:], None
        """ 
         
        HP["m"] = train_mX.shape[0]
        # train_mX, train_mY = train_X[n_data0:n_data,:,:,:], train_Y[n_data0:n_data,:]
        
        if setting == "E" or setting == "ED" or setting == "EDR":            
            HP["epochs"] = int(hpe0 * level[t])            
            for i in range(len(hp_list)):
                hp_list[i]["epochs"] = int(hpe0 * level[t])
            
        if setting == "ED_1" or setting == "EDR_1":
            HP["epochs"] = int(hpe0 * level[0] * (t+1) **(1/2))
            for i in range(len(hp_list)):
                hp_list[i]["epochs"] = int(hpe0 * level[0] * (t+1) **(1/2))
            # print("HP_epochs", HP["epochs"])
            
        # if setting == "P1":
        #    HP["epochs"] = int(hpe0 * level[t])
            
        print("setting:", setting)
        ep = HP["epochs"]
            
        if setting == "D" or setting == "E":
            n_t = int((n_sample / (n_data * n_level))  * n_init)
                     
        if setting == "ED" or setting == "EDR":
            n_t = int(((n_sample / (n_data * n_level))**2) * n_init)
        if setting == "ED_1" or setting == "EDR_1":
            n_t = int(((n_sample / (n_data * n_level))**(3/2)) * n_init)
        if setting == "P":
            n_t = n_init
            
        print("n_sample, n_data, n_level, n_init, n_t:", n_sample, n_data, n_level, n_init, n_t)          
        # print("HP_epochs, n_t", HP["epochs"], n_t)
        n_t_list.append(n_t)
        e_t_list.append(HP["epochs"])
        d_t_list.append(n_data)
        d0_t_list.append(n_data0)
        
        if update_method != None: # and n_t>2:# t<int(n_level/2):           
            hp_list, model_list, history_hp = model_update_sample(val_X, val_Y, model_list, hp_list, eva_val_list, eva_val_history_list, HHP, HPcv, n_t, update_rate, method = update_method, other_data = history_hp, adjusted = HHP["adjusted"])
        elif setting == "E" or setting == "D" or setting == "P" or setting == "EDR" or setting == "EDR_1":
            model_list, eva_list, hp_list = select(model_list, eva_val_list, eva_val_history_list, hp_list, n_t, adjusted = HHP["adjusted"])
        elif setting == "ED" or setting == "ED_1":
            eva_list, hp_list = select_1(eva_val_list, eva_val_history_list, hp_list, n_t, adjusted = HHP["adjusted"])
        else:
            print("no such setting!")
        
        
        for i in range(len(hp_list)):            
            hp_list[i]["time"] = t+1
        
        # print(eva_list)
        
        if setting == "E" or setting == "D" or setting == "P" or setting == "EDR" or setting == "EDR_1":
            model_list, eva_val_list, eva_test_list, hp_list, eva_val_history_list = model_revaluate(model_list, train_mX, train_mY, val_X, val_Y, test_X, test_Y, hp_list)
        if setting == "ED" or setting == "ED_1":
            model_list, eva_val_list, eva_test_list, hp_list, eva_val_history_list = model_revaluate_1(train_mX, train_mY, val_X, val_Y, test_X, test_Y, HP_list = hp_list)
        # model_update_list, eva_val_list, eva_test_list, HP_list
        min_history, ave_history = loss_history_process(eva_val_history_list, nepoch = ep)
        
        # print("len(min_history_full), len(min_history)", len(min_history_full), len(min_history))
        # print("min_history_full, min_history", min_history_full, min_history)    
        min_history_full = min_history_full + min_history
        ave_history_full = ave_history_full + ave_history
        # print("min_history_full", min_history_full)           
        # model_list, eva_list, hp_list = model_sample_init(train_mX, train_mY, val_X, val_Y, HP, HHP, HPcv)
        t = t + 1
        print("eva_list", eva_val_list)
        eva_val_min_list.append(min(eva_val_list))
        eva_test_min_list.append(min(eva_test_list))
        meva_val_list.append(np.mean(eva_val_list))
        meva_test_list.append(np.mean(eva_test_list))
        if t == n_level:
            break
    
    print("min_history_full, ave_history_full", min_history_full, ave_history_full)
    datetime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    rows = zip(eva_val_min_list, eva_test_min_list, meva_val_list, meva_test_list, n_t_list, e_t_list, d_t_list, d0_t_list)
    with open(datetime + fn, "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    
    fn = "_full_"+fn
    datetime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    rows_1 = zip(min_history_full, ave_history_full)
    with open(datetime + fn, "w") as f:
        writer = csv.writer(f)
        for row in rows_1:
            writer.writerow(row)
    
    try:
        index_min = np.argmin(eva_val_min_list)
        hp_list_min = hp_list[index_min]
        hp_list_min["epochs"] = 25
        # eva_min, model = model_apps(model_code, train_X, train_Y, val_X, val_Y, HP = hp_list_min, other_utils = other_utils)
        eva_val_final, eva_test_final, model, eva_val_history = model_main_cnn(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP = hp_list_min)
    
        fn_1 = fn + "_final_test"
        with open(datetime + fn_1, "w") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
    except:
        print("test hp fail!")
    
    return model_list, eva_val_min_list, eva_test_min_list, meva_val_list, meva_test_list, min_history_full, ave_history_full
