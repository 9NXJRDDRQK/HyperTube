def HP_search_rand(data, HP, HPcv, HHP, d_HP = None, parameters = None):

    val_cost_min = 1000
    HP0 = copy.deepcopy(HP)
    best_HP = copy.deepcopy(HP)
    best_para = copy.deepcopy(parameters)
    n_sample = HHP["n_sample"]
    para_iter = HHP["para_iter"]
    iteration = HHP["iteration"]
    flexible_act = False
    
    t0 = time.clock()
    cost_t = []
    cost_min_list = []
    HP_random = {}
    
    hp_opt_list = []
    val_cost_list = []
    eva_val_list = []
    eva_test_list = []
    
    for hp in HPcv.keys():

        if iteration==False or d_HP==None:
            r = HPcv[str(hp)]
            # HP_random[str(hp)] = np.random.uniform(low = r[0], high = r[1], size = n_sample)
            HP_random[str(hp)] = random.sample(population = set(r), k = n_sample)
            # HP_random[str(hp)] = np.random.choice(a = r, size = n_sample, replace = False, p=None)
        else:
            pp = d_HP[str(hp)]
            mu = pp[0]
            sigma = pp[1]
            r = HPcv[str(hp)]
            HP_random[str(hp)] = random.uniform(low = max(0, mu - sigma), high = (mu + sigma), size = n_sample) #*r

    # print("HP_random[str(hp)]", HP_random[str(hp)])
    for i in range(n_sample):

        HP = copy.deepcopy(HP0)

        if iteration == True:
            HP = copy.deepcopy(best_HP)

        para = None

        if para_iter==True:
            para = copy.deepcopy(best_para)

        print_dict = {}

        for hp in HPcv.keys():

            hp_rand = HP_random[str(hp)]
            HP[str(hp)] = hp_rand[i]

            print_dict[str(hp)] = HP[str(hp)]
        
        # print(print_dict)
        
        # print("HP", HP)
        # print("len(data)", len(data))
        
        # print("model(data, HP)", len(model(data, HP, model_in = None)))

        eva_val, eva_test, model = model_main_cnn(data = data, hparameters = HP, model_in = None)
            
        try:
            val_cost = eva_val.item()
            test_cost = eva_test.item()
        except:
            val_cost = eva_val
            test_cost = eva_test
            
        eva_val_list.append(val_cost)
        eva_test_list.append(test_cost)
            
        # val_cost = eva
        print("val_cost",val_cost)
        
        hp_opt_list.append(print_dict)
        val_cost_list.append(val_cost)

        if val_cost < val_cost_min:

            val_cost_min = val_cost
            best_para = copy.deepcopy(parameters)
            best_HP1 = copy.deepcopy(best_HP)
            best_HP = copy.deepcopy(HP)
            print("val_cost_min",val_cost_min)
            t = time.clock() - t0
            cost_t.append(t)
            cost_min_list.append(val_cost_min)
        else:
            continue

    """
    d_HP={}

    for hp in HPcv.keys():

        bhp = best_HP[str(hp)]
        dab = abs(best_HP[str(hp)] - best_HP1[str(hp)])
        d_HP[str(hp)] = [bhp, dab]
    """

    return eva_val_list, eva_test_list
