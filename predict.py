import numpy as np 
import matplotlib.pyplot as plt 
from defuzzification import Defuzzification
from utils import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from math import sqrt
from copy import deepcopy
from metrics import rrse
from preprocessing import Preprocess

def predict(Fuzzyfy, lags_used = [], num_groups=5, ndata=[''], data=[],in_sample=[],out_sample=[], lag = 0, mf_params_=[],num_series=[],agg_training='',yp_lagged='',h_prev=0,n_attempt=0,wd_=[],ensemble_antecedents=[],ensemble_rules=[],not_used_lag = False, detrend_series=False, diff_series=False,filepath='',lim=0, defuzz_method='cog',fig_axis=[3,2]):
    '''
    Module to time series forecasting. It uses multi-stepping in order to evaluate the model.
    INPUTS:
    \n - Fuzzyfy: Object containing informations about Fuzzification.
    \n - lags_used: If not_used_lags is true, masks series that isn't in list.
    \n - num_groups: Number of fuzzy sets.
    \n - ndata: name of data (e.g. column header)
    \n - data: data of the problem
    \n - in_sample: in_sample set of data
    \n - out_sample: out_sample set of data
    \n - lag:
    \n - mf_params: Membership function parameters
    \n - num_series: Number of series
    \n - agg_training: Aggregation terms in training set. Used to simplify deffuzification of training set.
    \n - yp_lagged
    \n - h_prev:
    #TODO - Continue this list

    VARIABLES:
    \n - y_predict_: Training set prediction
    \n - yp_totest: Input pattern to evaluate prediction
    \n - yt_totest: Output data, for each horizon and serie.
    '''
    
    defuzz = Defuzzification(mf_params_,num_series)

    y_predict_ = defuzz.run(defuzz_method,agg_training)

    yp_totest = yp_lagged[yp_lagged.shape[0]-1:yp_lagged.shape[0],:]
    yt_totest = np.zeros((h_prev,num_series))

    #Prediction - Multi-step
    for h_p in range(h_prev):

        #Check activated terms.
        mX_values_in = np.zeros((1,mf_params_.shape[0],yp_totest.shape[1]))
        antecedents_activated = []
        it = 0
        for i in range(num_series):
            mf_params = mf_params_[:,i]
            for j in range(lag):

                mX, _ = Fuzzyfy.fuzzify(np.array([yp_totest[0,i*lag+j]]),mf_params,num_groups=num_groups)
                mX_values_in[:,:,i*lag+j] = mX


                idx_nonzero = np.where(mX[0,:] > 0)
                idx_nonzero = idx_nonzero[0]

                if not_used_lag:
                    for k in range(idx_nonzero.shape[0]):
                        if j in lags_used[i]:
                            antecedents_activated.append((it,idx_nonzero[k]))
                        else:
                            pass
                    it += 1
                
                else:
                    for k in range(idx_nonzero.shape[0]):
                        antecedents_activated.append((i*lag+j,idx_nonzero[k]))

        '''
        if not_used_lag:
            mX_values_in, _ = remove_lags(mX_values_in,lag_notused,num_series,lag)


        prem_terms_test = np.zeros((ensemble_antecedents.shape[0],1))
        '''
        rules_idx = []
        check_idx = 0
        
        #Checking for every rule in dataset if it's activated
        #TODO - Check if we can modify this into enumerate, avoiding check_idx += 1 every time.
        for n_rule in ensemble_antecedents:
            #print('Rule {} is {}'.format(check_idx,test(n_rule,antecedents_activated)))
            if test(n_rule,antecedents_activated):
                rules_idx.append(check_idx)
            check_idx += 1
            
        prem_activated = np.zeros((ensemble_antecedents.shape[0],))
        for i in rules_idx:
            prem_activated[i,] = prem_term(ensemble_antecedents[i,0],mX_values_in)
        
        agg_test = np.zeros((wd_.shape))
        for i in range(num_series):
            for j in rules_idx:
                rule = ensemble_rules[j,i]
                consequent = rule[-1]
                agg_test[j,consequent[1],i] = prem_activated[j,]
                
                
        weight_agg = np.multiply(agg_test,wd_)
        weight_ = np.zeros((weight_agg.shape[1],weight_agg.shape[2]))

        for i in range(weight_.shape[1]):
            weight_[:,i] = weight_agg[:,:,i].max(axis=0)

        w_todefuzz = np.reshape(weight_,(1,weight_.shape[0],weight_.shape[1]))
        
        #Defuzzification in fact
        y_pred = defuzz.run(defuzz_method,w_todefuzz,show=False)
        
        #Store predicted value into yt_totest.
        yt_totest[h_p,:] = y_pred
        
        #Last step, we use the predicted output to compose input data.
        y_temp = np.zeros(yp_totest.shape)
        assert y_temp.shape == yp_totest.shape
        y_temp[0,1:] = yp_totest[0,0:yp_totest.shape[1]-1]
        for ii in range(num_series):
            #print(yp_totest[0,ii*lag])
            #print(y_pred[0][ii])
            #yp_totest[0,ii*lag] = y_pred[0][ii]
            y_temp[0,ii*lag] = y_pred[0][ii]
            #print(yp_totest[0,yp_totest.shape[1]-1])
        yp_totest = y_temp

    #Plot training results
    #plot_training(y_predict_=y_predict_,num_series=num_series,in_sample=in_sample,lag=lag,ndata=ndata,data=data,trends=[],filename='{}/Insample {}'.format(filepath,n_attempt),fig_axis=fig_axis)

    #Plot predicted results and returns error metrics
    errors = plot_predict(lim=lim,yt_totest=yt_totest,num_series=num_series,data=data,in_sample=in_sample,out_sample=out_sample,trends=[],ndata=ndata,filename='{}/Outsample {}'.format(filepath,n_attempt), fig_axis=fig_axis)

    return errors

def plot_training(y_predict_=[],num_series=0,in_sample=[],lag=0,ndata=[],data=[],trends=[],filename=[], diff_series=False, detrend_series=False,fig_axis=[3,2]):
    y_predict_new = np.ndarray(shape=y_predict_.shape)
    '''
    for i in range(num_series):
        y_predict__ = y_predict_[:,i]
        y_predict__[y_predict__ < 0.1] = np.nan
        
        idx = np.where(np.isnan(y_predict__))
        if len(idx) > 0:
            y_predict_new[:,i] = pd.DataFrame(y_predict__).fillna(method='bfill',limit=9000).values.ravel()
            #print(i)
        else:
            y_predict_new[:,i] = y_predict_[:,i]
    '''

    #TODO - Verificar se o Yt__ esta certo para diff_series e detrend
    y_predict_new = deepcopy(y_predict_)
    if diff_series:
        Y__ = y_predict_new + data[lag:in_sample.shape[0]-1,:]
        Yt__ = in_sample[lag+1:] + data[lag:in_sample.shape[0]-1,:]

    elif detrend_series:
        Y__ = y_predict_new + trends[lag:in_sample.shape[0]-1,:]
        Yt__ = in_sample[lag+1:] + trends[lag:in_sample.shape[0]-1,:]

    else:
        Y__ = y_predict_new
        #Yt__ = in_sample[lag+1:]
        Yt__ = in_sample


    plt.figure(figsize=(16*2,10*3))
    k = 1
    for i in range(num_series):
        plt.subplot(fig_axis[0],fig_axis[1],k)
        plt.title('Serie {}'.format(ndata.columns[i]),fontsize=30)
        plt.plot(Y__[:,i],color='blue')
        plt.plot(Yt__[:,i],color='red')
        plt.legend(['Predicted','Target'])
        plt.xlabel('Time(h)',fontsize=15)
        plt.ylabel('Value',fontsize=15)
        k += 1
    plt.savefig('results/{}.png'.format(filename))
    #plt.show()
    plt.close()

def plot_predict(lim=0,yt_totest=[],num_series=0,data=[],in_sample=[],out_sample=[],trends=[],ndata=[],filename='', diff_series=False, detrend_series=False,fig_axis=[3,2]):
    #yt_totest = yt_totest[:9,:]
    #out_sample = out_sample[:9,:]

    for i in range(num_series):
        idx = np.where(np.isnan(yt_totest[:,i]))

        if len(idx) > 0:
            yt_totest[:,i] = pd.DataFrame(yt_totest[:,i]).fillna(method='bfill').values.ravel()

    if diff_series:
        #Y__ = yt_totest + data[in_sample.shape[0]:data.shape[0]-1,:]
        #Yt__ = out_sample + data[in_sample.shape[0]:data.shape[0]-1,:]
        y_pp = np.roll(yt_totest,1,axis=0)
        y_pp[0,:] = data[in_sample.shape[0],:]
        y_tt = np.roll(out_sample,1,axis=0)
        y_tt[0,:] = data[in_sample.shape[0],:]
        Y__ = yt_totest + y_pp
        Yt__ = out_sample + y_tt
        print('diff series')

    elif detrend_series:
        Y__ = yt_totest + trends[in_sample.shape[0]:,:]
        Yt__ = out_sample + trends[in_sample.shape[0]:,:]

    else:
        Y__ = yt_totest
        Yt__ = out_sample
    errors = np.zeros(shape=(2,num_series))
    rmse = sqrt(mean_squared_error(Yt__[:,0], Y__[:,0]))
    mae = mean_absolute_error(Yt__[:,0], Y__[:,0])
    errors[0,0] = rmse
    errors[1,0] = mae
    if rmse < lim:
        with open('results/{}.txt'.format(filename),'w') as f:
            for i in range(num_series):
                rmse = sqrt(mean_squared_error(Y__[:,i], Yt__[:,i]))
                mae = mean_absolute_error(Y__[:,i], Yt__[:,i])
                rrse_error = rrse(Y__[:,i], Yt__[:,i])
                
                print('Outsample RRSE for serie {} is {} \n'.format(i+1,rrse_error), file=f)
                print('Outsample RMSE for serie {} is {} \n'.format(i+1,rmse), file=f)
                print('Outsample MAE for serie {} is {} \n'.format(i+1,mae), file=f)
                print('Outsample SMAPE for serie {} is {} \n'.format(i+1,smape(Yt__[:,i],Y__[:,i])),file=f)
                errors[0,i] = rmse
                errors[1,i] = rrse_error

    plt.figure(figsize=(16*3,10*2))
    k = 1
    for i in range(num_series):
        plt.subplot(fig_axis[0],fig_axis[1],k)
        plt.title('Serie {}'.format(ndata.columns[i]),fontsize=30)
        plt.plot(Y__[:,i],color='blue')
        plt.plot(Yt__[:,i],color='red')
        plt.legend(['Predicted','Target'])
        plt.xlabel('Time(h)',fontsize=15)
        plt.ylabel('Value',fontsize=15)
        k += 1
    
    if errors[0,0] < lim:
        plt.savefig('results/{}.png'.format(filename))    #plt.show()
    plt.close()

    return errors



def predict_pattern(Fuzzyfy, lags_used = [], num_groups=5, ndata=[''], data=[], lag = 0, mf_params_=[],num_series=[],h_prev=0,n_attempt=0,wd_=[],not_used_lag = False, detrend_series=False, diff_series=False,filepath='',lim=0, defuzz_method='cog',fig_axis=[3,2], n_patterns=0, list_rules=None):
    '''
    Function for pattern prediction. 
    \n Important features: 
    \n - n_pattern: Number of seasonal patterns.
    \n - list_rules: list of rules for each pattern.
    '''

    preprocess_data = Preprocess(data,h_prev=h_prev,num_series=num_series)
    
    in_sample, out_sample = preprocess_data.split_data()
    
    yt, yp, yp_lagged = preprocess_data.delay_input(in_sample = in_sample, lag = lag)


    defuzz = Defuzzification(mf_params_,num_series)

    yp_totest = yp_lagged[yp_lagged.shape[0]-1:yp_lagged.shape[0],:]
    yt_totest = np.zeros((h_prev,num_series))

    for h_p in range(h_prev):
        print('='*89)
        #Select which ruleset use now.

        rem = h_p % 168

        k = rem // 24

        print(f'Debug only, rem = {rem} and k = {k}')
        print('='*89)
        ensemble_antecedents = list_rules[k].rules
        ensemble_rules = list_rules[k].complete_rules

        #Just faking an weighted rule as ones.
        wd_ = np.ones((ensemble_rules.shape[0], num_groups, ensemble_rules.shape[1]))


        print(f'Shape of ensemble rules is {ensemble_rules.shape}')

        mX_values_in = np.zeros((1,mf_params_.shape[0],yp_totest.shape[1]))
        antecedents_activated = []
        it = 0
        for i in range(num_series):
            mf_params = mf_params_[:,i]
            for j in range(lag):

                mX, _ = Fuzzyfy.fuzzify(np.array([yp_totest[0,i*lag+j]]),mf_params,num_groups=num_groups)
                mX_values_in[:,:,i*lag+j] = mX


                idx_nonzero = np.where(mX[0,:] > 0)
                idx_nonzero = idx_nonzero[0]

                if not_used_lag:
                    for k in range(idx_nonzero.shape[0]):
                        if j in lags_used[i]:
                            antecedents_activated.append((it,idx_nonzero[k]))
                        else:
                            pass
                    it += 1
                
                else:
                    for k in range(idx_nonzero.shape[0]):
                        antecedents_activated.append((i*lag+j,idx_nonzero[k]))

        '''
        if not_used_lag:
            mX_values_in, _ = remove_lags(mX_values_in,lag_notused,num_series,lag)


        prem_terms_test = np.zeros((ensemble_antecedents.shape[0],1))
        '''
        rules_idx = []
        check_idx = 0
        
        #Checking for every rule in dataset if it's activated
        #TODO - Check if we can modify this into enumerate, avoiding check_idx += 1 every time.
        for n_rule in ensemble_antecedents:
            #print('Rule {} is {}'.format(check_idx,test(n_rule,antecedents_activated)))
            if test(n_rule,antecedents_activated):
                rules_idx.append(check_idx)
            check_idx += 1
            
        prem_activated = np.zeros((ensemble_antecedents.shape[0],))
        for i in rules_idx:
            prem_activated[i,] = prem_term(ensemble_antecedents[i,0],mX_values_in)
        
        agg_test = np.zeros((wd_.shape))
        for i in range(num_series):
            for j in rules_idx:
                rule = ensemble_rules[j,i]
                consequent = rule[-1]
                agg_test[j,consequent[1],i] = prem_activated[j,]
                
                
        weight_agg = np.multiply(agg_test,wd_)
        weight_ = np.zeros((weight_agg.shape[1],weight_agg.shape[2]))

        for i in range(weight_.shape[1]):
            weight_[:,i] = weight_agg[:,:,i].max(axis=0)

        w_todefuzz = np.reshape(weight_,(1,weight_.shape[0],weight_.shape[1]))
        
        #Defuzzification in fact
        y_pred = defuzz.run(defuzz_method,w_todefuzz,show=False)
        
        #Store predicted value into yt_totest.
        yt_totest[h_p,:] = y_pred
        
        #Last step, we use the predicted output to compose input data.
        y_temp = np.zeros(yp_totest.shape)
        assert y_temp.shape == yp_totest.shape
        y_temp[0,1:] = yp_totest[0,0:yp_totest.shape[1]-1]
        for ii in range(num_series):
            #print(yp_totest[0,ii*lag])
            #print(y_pred[0][ii])
            #yp_totest[0,ii*lag] = y_pred[0][ii]
            y_temp[0,ii*lag] = y_pred[0][ii]
            #print(yp_totest[0,yp_totest.shape[1]-1])
        yp_totest = y_temp


    errors = plot_predict(lim=lim,yt_totest=yt_totest,num_series=num_series,data=data,in_sample=in_sample,out_sample=out_sample,trends=[],ndata=ndata,filename='{}/Outsample {}'.format(filepath,n_attempt), fig_axis=fig_axis)

    return errors


            