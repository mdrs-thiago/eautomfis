from utils import nmean_activation, check_if_inside, check_duplicate_rules, rearranje_rules, freq_activation, mean_activation
from tnorm import tnorm_product, tnorm_minimum
import numpy as np
from copy import deepcopy
from math import sqrt

class Formulation():

    def __init__(self,R,min_activation,method):
        self.max_antecedents = R
        self.min_activation = min_activation
        self.method = method

    def run(self,mX_lagged_):
        #print(f'Using {self.method} on formulation')
        return self.nonExaustive(self,mX_lagged_)

    @staticmethod
    def nonExaustive(self, mX_lagged_):
        rulesize = [0]
        prem_terms = np.array([])
        for r in range(self.max_antecedents):
            if r == 0:
                rules1 = []        
                for i in range(mX_lagged_.shape[2]):
                    for j in range(mX_lagged_.shape[1]):
                        #print(mX_lagged_[:,j,i].shape)
                        if self.method is 'nmean':
                            activation, _, _ = nmean_activation(mX_lagged_[:,j,i],self.min_activation)
                        elif self.method is 'freq':
                            activation, _, _ = freq_activation(mX_lagged_[:,j,i],self.min_activation)
                        elif self.method is 'mean':
                            activation, _, val = mean_activation(mX_lagged_[:,j,i],self.min_activation)
                            #print(f'{mX_lagged_[:,j,i].mean()} and {val}')
                            #assert (mX_lagged_[:,j,i].mean() == val), 'Values mismatch'
                        else:
                            print('Method not found')
                        #print(mean_activation)
                        #print(freq)
                        #print(activation)
                        if activation is True:
                            rules1.append([(i,j)])
                            if prem_terms.size == 0:
                                prem_terms = mX_lagged_[:,j,i]
                            else:
                                prem_terms = np.vstack((prem_terms,mX_lagged_[:,j,i]))
                            
                rules = np.empty(shape=[len(rules1), 1],dtype='object')
                rulesize.append(len(rules1))
                rules[:,0] = rules1
            
            else:
                lim_sup = rulesize[r]
                lim_inf = rulesize[r-1]
                new_rules = []           #Reinicia a lista de novas regras a cada layer de regra.
                #print(lim_sup,lim_inf)
                #Vamos verificar cada regra criada na rodada anterior. Para isso, verificamos o range que a lista de regras esta alocada
                for rule in range(lim_inf,lim_inf+lim_sup):
                    
                    grow_rule = rules[rule,0]
                    #print(grow_rule)
                    for i in range(mX_lagged_.shape[2]):
                        for j in range(mX_lagged_.shape[1]):
                            
                            
                            #Checa se o novo antecedente ja esta dentro do conjunto de antecedentes da regra
                            if check_if_inside((i,j),grow_rule):
                                continue
                            #Vamos concatenar todas as regras
                            count_tnorm = mX_lagged_[:,j,i]
                            
                            for r_size in grow_rule:               
                                count_tnorm = np.vstack((count_tnorm,mX_lagged_[:,r_size[1],r_size[0]]))
                                
                                #print(count_tnorm.shape)
                                #print(count_tnorm[:,1:4])
                            tnorm_ = tnorm_product(count_tnorm)
                            #print(tnorm_min[1:4])
                            if self.method is 'nmean':
                                #activation, _, _ = nmean_activation(tnorm_,self.min_activation)
                                activation, _, val = nmean_activation(tnorm_,self.min_activation)
                            elif self.method is 'freq':
                                #activation, _, _ = freq_activation(tnorm_,self.min_activation/sqrt(r))
                                activation, _, val = freq_activation(tnorm_,self.min_activation)
                            elif self.method is 'mean':
                                #activation, _, _ = mean_activation(tnorm_,self.min_activation/sqrt(r))
                                activation, _, val = mean_activation(tnorm_,self.min_activation)
                                #if (val > self.min_activation):
                                    #print(f'Mean is {val} and tnorm_mean is {tnorm_.mean()}')
                                    #print(tnorm_)
                            else:
                                print('Method not found')

                            if activation is True:
                                rule_to_append = deepcopy(grow_rule)
                                
                                rule_to_append.append((i,j))
                                #print(rule_to_append)
                                #print(mean_activation)
                                #print(freq)
                                sorted_rule = rearranje_rules(rule_to_append)
                                #print(f'Added {sorted_rule} to base rule with activation = {val}')
                                #print(sorted_rule) 
                                if not check_duplicate_rules(sorted_rule, new_rules):
                                    new_rules.append(sorted_rule)
                                    prem_terms = np.vstack((prem_terms,tnorm_))

                                #else:
                                    #print('Found one')
                                    

                                
                rulesize.append(len(new_rules))
                
                rules_ = np.empty(shape=[len(new_rules), 1],dtype='object')
                rules_[:,0] = new_rules
                rules = np.concatenate((rules,rules_))


        return rules, rulesize, prem_terms



