import numpy as np
import operator
import random
import time

def roulette(eva_list):
    
    l = len(eva_list)
    
    eva_list_1 = [(np.max(eva_list)-eva_list[i]) for i in range(l)]
    s_eva = np.sum(eva_list_1)
    prob_list = [eva_list_1[i]/s_eva for i in range(l)]
    
    return prob_list 

def crossover_hp(HPga, hp_individual_1, hp_individual_2, prob_cross):

    individual_1 = encoding(HPga, hp_individual_1)
    individual_2 = encoding(HPga, hp_individual_2)
    
    for i in range(len(individual_1)):

        rand = random.uniform(0,1)
        if rand < prob_cross:
            print("cross!")
            print("individual_1[i]",individual_1[i])
            segment1 = individual_1[i]
            segment2 = individual_2[i]
            individual_1[i] = segment2
            individual_2[i] = segment1
            print("segment1", segment1, "segment2", segment2)
            print("individual_1[i]",individual_1[i])
    
    new_individual_1 = decoding(HPga, individual_1)
    new_individual_2 = decoding(HPga, individual_2)
    
    for hp in new_individual_1.keys():
        hp_individual_1[str(hp)] = new_individual_1[str(hp)]
        hp_individual_2[str(hp)] = new_individual_2[str(hp)]
    
    return hp_individual_1, hp_individual_2


def mutation_hp(hp_individual, prob, HPga):
    # print("hp_individual", hp_individual)
    individual = encoding(HPga, hp_individual)
    # print("individual", individual)
    # print("code", individual)
    individual_new = mutation(individual, prob)
    # print("code_new", individual_new)
    # print("individual_new", individual_new)
    new_individual = decoding(HPga, individual_new)
    # print("new_individual", new_individual)
    
    for hp in new_individual.keys():
        hp_individual[str(hp)] = new_individual[str(hp)]
        
    # print("hp_individual_new", hp_individual)
    return hp_individual

def mutation(individual, prob):

    l = len(individual)
    individual1 = []

    for i in range(l):
        lhp = len(individual[i])
        individual_hp = individual[i]
        # for j in range(int(len(individual_hp)/2), len(individual_hp)):
        for j in range(np.min([1, int(len(individual_hp)/3)]), len(individual_hp)):
            if random.uniform(0,1) < prob:
                individual_hp[j]=1-individual_hp[j]
        individual1.append(individual_hp)

    return individual

def encoding(HPga, HP1):

    HPga_code=[]
    # print("HP1", HP1)

    for hp in HPga.keys():

        hp_range = HPga[str(hp)]
        # print("hp", hp)
        # print("HP1[str(hp)]", HP1[str(hp)])
        # print("hp_range", hp_range)
        l0 = len(hp_range)-1
        l = l0

        lhp = len(dicimal_to_binary(l0))
        hp1_code_0 = [0]*lhp
        for j in range(l0+1):
            # print("HP1[str(hp)]", HP1[str(hp)])
            # print("hp_range[j]", hp_range[j])
            if(HP1[str(hp)]==hp_range[j]):
                # print("haha")
                hp1_code = dicimal_to_binary(j)
        
        hp1_code.reverse()
        hp1_code.extend([0]*(lhp-len(hp1_code)))
        hp1_code.reverse()
        HPga_code.append(hp1_code)

    return HPga_code

def decoding(HPga, HPga_code):

    HP1 = {}

    i = 0
    for hp in HPga.keys():
        j = binary_to_dicimal(HPga_code[i])
        # print("hp:",hp,"j:",j)
        hp_val = HPga[str(hp)]
        HP1[str(hp)] = hp_val[j]
        i = i+1

    return HP1

def dicimal_to_binary(val):

    l = val
    bin_val=[]

    while l>0:
        m = l%2
        l = int(l/2)
        bin_val.append(m)

    bin_val.reverse()

    return bin_val

def binary_to_dicimal(bin_val):

    bin_val.reverse()

    dic_val=0
    l = len(bin_val)

    for i in range(l):
        dic_val = dic_val + (2**i)*bin_val[i]

    return dic_val
