import numpy as np
from tnorm import tnorm_product

#Metrics
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def mape(A,F):
    return np.mean(np.abs(np.divide(A-F,A)))


#Useful functions

def test(rule,antecedents_activated):
    for term in rule[0]:
        
        if term in antecedents_activated:
            pass
        else:
            return False
    
    return True


def prem_term(rule,muX):
    prem_concat = []
    for term in rule:
        #print(term)
        prem_concat.append(muX[0,term[1],term[0]])
    return tnorm_product(prem_concat)


#Function to check if the antecedent is inside a function

def check_if_inside(val,eachRule):
    if val in eachRule:
        return True
    
    return False


#Function just to sort antecedent rules in numerical order.

def rearranje_rules(rule):
    sorted_rule = []
    first_rules = np.unique([num_series[0] for num_series in rule])
    
    for val in first_rules:
        arranje_rule = [thisantecedent for thisantecedent in rule if thisantecedent[0] == val]
        arranje_rule.sort(key=lambda x: x[1])
        
        sorted_rule.extend(arranje_rule)
        
    return sorted_rule

#Function to detect duplicated rules.

def check_duplicate_rules(val,rules):
    for rule in rules:
        if val == rule:
            return True
    return False



def nmean_activation(values,min_act):
    values = np.asarray(values)
    if np.sum(values) == 0:
        return False, 0, 0
    else:
        mean_activation = values[values != 0].mean()

        check_activation = mean_activation
        if check_activation > min_act:
            activation = True
        else:
            activation = False
        return activation, mean_activation, check_activation
    
def freq_activation(values,min_act):
    values = np.asarray(values)
    if np.sum(values) == 0:
        return False, 0, 0
    else:

        freq_rel = values[values != 0].shape[0]/values.shape[0]

        check_activation = freq_rel
        if check_activation > min_act:
            activation = True
        else:
            activation = False
        return activation, freq_rel, check_activation

def mean_activation(values,min_act):
    values = np.asarray(values)
    if np.sum(values) == 0:
        return False, 0, 0
    else:
        mean_activation = values.mean()

        check_activation = mean_activation
        if check_activation > min_act:
            activation = True
        else:
            activation = False
        return activation, mean_activation, check_activation


def find_rules_by_consequent(rules,n_serie,n_set):
    index_ = []
    k = 0
    for rule in rules[:,n_serie]:
        if rule[-1] == (n_serie,n_set):
            index_.append(k)
        k += 1
    #print(index_)
    return index_