#Basic imports
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from math import sqrt
from copy import deepcopy

#Imports from sklearn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

#autoMFIS modules imports
from fuzzyfication import Fuzzification
from tnorm import tnorm_product
from formulation import Formulation
from split import Split
from reweight import Reweight
from defuzzification import Defuzzification
from metrics import mape, smape
from preprocessing import Preprocess
from basicfuzzy import trimf, trapmf
from utils import *
from metrics import rrse
from filter import Filter

#Predict imports
from predict import plot_predict, plot_training

#Some functions.
#TODO - Check if these functions are already on utils module.

def remove_lags(mX_lagged_,lag_notused,num_series,lag):
    assert num_series == lag_notused.shape[0]
    lags_used = np.array(lag_notused)
    for n in range(num_series):
        lag_serie = lag_notused[n]
        lin = np.linspace(0,lag-1,lag)
        lag_used = np.setdiff1d(lin,lag_serie) + n*lag
        lag_used = [int(f) for f in lag_used]
        lags_used[n] = lag_used
        print(lag_used)
        if n == 0:
            new_mX = mX_lagged_[:,:,lag_used[:]]   
            #print(new_mX.shape) 
        else:
            new_mX = np.concatenate((new_mX,mX_lagged_[:,:,lag_used]),axis=2)
    
    return new_mX, lags_used


class autoMFIS():
    '''
    Automatic Fuzzy System for multivariate time series forecasting.
    
    \n Parameters are divided into methods, booleans and numeric parameters.
    \n Methods:
    \n - fuzzy_method: Method for fuzzy set generation. Options: 'mfdef_triangle' (uniform division of fuzzy sets), 'mfdef_cluster' (division of centers by clustering method).
    \n - form_method: Method for formulation evaluation. Options: 'freq' (frequency), 'mean' (mean of activation), 'nmean' (non-zero mean of activation).
    \n - solve_method: Method for optimization problem. Options: 'None' (skip reweight), 'mqr' (Constrained optimization).
    \n - defuzz_method: Method for defuzzification. Options: 'cog' (center of gravity), 'mom' (mean of maximum), 'height' (height method).
    
    \n Boolean parameters:
    \n - diff_series: If true, use diff series preprocessing.
    \n - detrend_series: If true, use detrend series preprocessing.
    \n - hide_values: If true, mask some desired series.

    \n Numeric parameters:
    \n - num_groups: Number of group sets during fuzzification.
    \n - h_prev: Prediction horizon, relative to the problem.
    \n - num_series: Number of input series.
    \n - min_activation: Min. activation to be elected as antecedent rule.
    \n - max_rulesize: Max. of antencedents
    \n - lag: Max. lags.
    '''
    def __init__(self,diff_series=False,detrend_series=False,fuzzy_method='mfdef_triangle',solve_method='None',defuzz_method='mom', num_groups = 5,
        h_prev = 1, num_series = 1, max_rulesize = 5, min_activation = 0.5, lag = 1, hide_values = False, form_method = 'nmean', show=False, split_method = 'FCD'):
        self.diff_series = diff_series
        self.detrend_series = detrend_series

        self.fuzzy_method = fuzzy_method
        self.solve_method = solve_method 
        self.defuzz_method = defuzz_method
        self.num_groups = num_groups
        self.form_method = form_method
        self.hide_values = hide_values
        self.split_method = split_method

        self.h_prev = h_prev
        self.num_series = num_series 
        self.max_rulesize = max_rulesize
        self.min_activation = min_activation
        self.lag = lag 
        self.show = show

        self.set_fuzzy = False

    def set_fuzzification(self, Fuzzify, mf_params, mX, mY, mX_lagged):
        self.set_fuzzy = True
        self.Fuzzify = Fuzzify
        self.mf_params = mf_params
        self.mX = mX
        self.mY = mY
        self.mX_lagged = mX_lagged

    def train(self, data, yt=None,yp=[],yp_lagged=[],in_sample=None,out_sample=[],not_select_subsample=[], lag_notused=[],debug=False):
        '''
        Training step for autoMFIS. It's divided into 6 steps, namely:
        \n - 0. Preprocessing (if not given)
        \n - 1. Fuzzification
        \n - 2. Formulation
        \n - 3. Splitting
        \n - 4. Reweight
        \n - 5. Defuzzification
        '''
        #Preprocessing
        if in_sample is None:
            print('In-sample not given to autoMFIS module. Running preprocessing...')
            prep = Preprocess(data, h_prev = self.h_prev, num_series = self.num_series)
            in_sample, out_sample = prep.split_data()
            yt, yp, yp_lagged = prep.delay_input(in_sample = in_sample, lag = self.lag)


        #Fuzzificacao
        #Lembrete: 
        #axis 0 - Registros da série
        #axis 1 - Valor de pertinência ao conjunto Fuzzy
        #axis 2 - Numero de séries
        if debug:
            print('Step 1 - Fuzzification')
        
        Fuzzify, mf_params_, mX_, mY_, mX_lagged_ = self.fuzzify(in_sample, yp, yt, yp_lagged, not_select_subsample)
        
        #assert (mX_lagged_[:,:,not_select_subsample] == 0).all(), "Cant hide subsample"

        #print(np.unique(mX_))
        #print(np.unique(mY_))
        #print(np.unique(mX_lagged_))
        #print(mX_lagged_[:,:,not_select_subsample])
        ############## Formulacao
        
        if self.hide_values:
            new_mX, _lags_used = remove_lags(mX_lagged_,lag_notused,self.num_series,self.lag)

        else:
            new_mX = mX_lagged_
        
        #print(np.unique(new_mX))
        #Formulation
        if debug:
            print('Step 2 - Formulation')
        rules, prem_terms = self.formulate(new_mX)

        #Splitting method
        if debug:
            print('Step 3 - Split')
        complete_rules = self.split_(mY_, prem_terms, rules)

        if debug:
            print('Step 4 - Filter')
        
        complete_rules, prem_terms, rules = self.filtering(complete_rules, prem_terms, rules)

        ############## Reweight
        if debug:
            print('Step 5 - Reweight')

        wd_, agg_training = self.reweight_mf(mY_,complete_rules,prem_terms)

        ############## Defuzzification
        if debug:
            print('Step 6 - Defuzzification')
        self.defuzzify(mf_params_, agg_training)

        return complete_rules, prem_terms, rules, agg_training, wd_
    


    def fuzzify(self,in_sample, yp, yt, yp_lagged, not_select_subsample):

        if self.set_fuzzy:
            Fuzzify = self.Fuzzify
            mf_params_ = self.mf_params

            mX_ = self.mX
            mY_ = self.mY
            mX_lagged_ = self.mX_lagged

            #mX_lagged_[:,:,not_select_subsample] = 0


        else:
            print('Fuzzy not given')
            Fuzzify = Fuzzification(self.fuzzy_method)
            self.Fuzzify = Fuzzify
            first_time = True
            for n in range(self.num_series):
                
                _, mf_params = Fuzzify.fuzzify(in_sample[:,n],np.array([]),num_groups=self.num_groups)
                mX, _ = Fuzzify.fuzzify(yp[:,n],mf_params,num_groups=self.num_groups)
                mY, _ = Fuzzify.fuzzify(yt[:,n],mf_params,num_groups=self.num_groups)
                if first_time:
                    mX_ = np.ndarray([mX.shape[0],mX.shape[1], self.num_series])
                    mY_ = np.ndarray([mY.shape[0],mY.shape[1], self.num_series])
                    mf_params_ = np.ndarray([mf_params.shape[0], self.num_series])
                    first_time = False
                mX_[:,:,n] = mX
                mY_[:,:,n] = mY
                mf_params_[:,n] = mf_params.ravel()
                #print(mf_params)
                #print(mX.shape)

            #self.mf_params = mf_params
            #self.mf_params_ = mf_params_
            self.mX_ = mX_
            self.mY_ = mY_ 

            mX_lagged_ = np.ndarray([mX_.shape[0],mX_.shape[1],yp_lagged.shape[1]])
            for i in range(self.num_series):
                mf_params = mf_params_[:,i]
                for j in range(self.lag):
                    mX, _ = Fuzzify.fuzzify(yp_lagged[:,i*self.lag+j],mf_params,num_groups=self.num_groups)
                    mX_lagged_[:,:,i*self.lag+j] = mX
                    #print(i*lag+j)
            mX_lagged_[:,:,not_select_subsample] = 0




        return Fuzzify, mf_params_, mX_, mY_, mX_lagged_

    def formulate(self, new_mX):
        form = Formulation(self.max_rulesize,self.min_activation,self.form_method)
        rules, _, prem_terms = form.run(new_mX)
        return rules, prem_terms

    def split_(self, mY_, prem_terms, rules):
        split = Split(mY_,prem_terms,self.num_series)
        complete_rules = split.run(rules, min_activation = self.min_activation, method=self.split_method)
        return complete_rules

    def filtering(self, complete_rules, prem_terms, rules):
        filt = Filter( prem_terms, complete_rules, rules)
        filtered_rules, filtered_prems, filtered_antecedents = filt.run(s=0.6)
        return filtered_rules, filtered_prems, filtered_antecedents

    def reweight_mf(self, mY_,complete_rules,prem_terms):
        rw = Reweight(mY_,complete_rules,prem_terms)
        wd_, agg_training = rw.run(self.solve_method,debug=False)

        return wd_, agg_training

    def defuzzify(self, mf_params_, agg_training):
        defuzz = Defuzzification(mf_params_,self.num_series)
        _y_predict = defuzz.run(self.defuzz_method,agg_training,show=self.show)


    def predict(self, initial_values, lags_used = [], ndata=[''], data=[], in_sample=[], out_sample=[], agg_training=None,h_prev=0,n_attempt=0,wd_=[],ensemble_antecedents=[],ensemble_rules=[],not_used_lag = False, filepath='',lim=0, fig_axis=[3,2], show = False, plot_image = True):
        '''
        Module to time series forecasting. It uses multi-stepping in order to evaluate the model.
        INPUTS:
        \n - Fuzzify: Object containing informations about Fuzzification.
        \n - lags_used: If not_used_lags is true, masks series that isn't in list.
        \n - num_groups: Number of fuzzy sets.
        \n - ndata: name of data (e.g. column header)
        \n - data: data of the problem
        \n - in_sample: in_sample set of data
        \n - out_sample: out_sample set of data
        \n - lag:
        \n - mf_params: Membership function parameters
        \n - agg_training: Aggregation terms in training set. Used to simplify deffuzification of training set.
        \n - yp_lagged
        \n - h_prev:
        #TODO - Continue this list

        VARIABLES:
        \n - y_predict_: Training set prediction
        \n - yp_totest: Input pattern to evaluate prediction
        \n - yt_totest: Output data, for each horizon and serie.
        '''
        
        defuzz = Defuzzification(self.mf_params,self.num_series)
        if agg_training is not None:
            y_predict_ = defuzz.run(self.defuzz_method,agg_training)

        yp_totest = initial_values
        yt_totest = np.zeros((h_prev,self.num_series))

        #Prediction - Multi-step
        for h_p in range(h_prev):

            #Check activated terms.
            mX_values_in = np.zeros((1,self.mf_params.shape[0],yp_totest.shape[1]))
            antecedents_activated = []
            it = 0
            for i in range(self.num_series):
                mf_params = self.mf_params[:,i]
                for j in range(self.lag):
                    mX, _ = self.Fuzzify.fuzzify(np.array([yp_totest[0,i*self.lag+j]]),mf_params,num_groups=self.num_groups)
                    mX_values_in[:,:,i*self.lag+j] = mX

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
                            antecedents_activated.append((i*self.lag+j,idx_nonzero[k]))

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
            for i in range(self.num_series):
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
            y_pred = defuzz.run(self.defuzz_method,w_todefuzz,show=show)
            
            #Store predicted value into yt_totest.
            yt_totest[h_p,:] = y_pred
            
            #Last step, we use the predicted output to compose input data.
            y_temp = np.zeros(yp_totest.shape)
            assert y_temp.shape == yp_totest.shape
            y_temp[0,1:] = yp_totest[0,0:yp_totest.shape[1]-1]
            for ii in range(self.num_series):
                #print(yp_totest[0,ii*lag])
                #print(y_pred[0][ii])
                #yp_totest[0,ii*lag] = y_pred[0][ii]
                y_temp[0,ii*self.lag] = y_pred[0][ii]
                #print(yp_totest[0,yp_totest.shape[1]-1])
            yp_totest = y_temp

        #Plot training results
        #plot_training(y_predict_=y_predict_,num_series=self.num_series,in_sample=in_sample,lag=self.lag,ndata=ndata,data=data,trends=[],filename='{}/Insample {}'.format(filepath,n_attempt),fig_axis=fig_axis)
        '''
        plt.figure(figsize=(16,10))
        
        for i in range(self.num_series):
            plt.subplot(fig_axis[0],fig_axis[1],i+1)
            #plt.title('Serie {}'.format(ndata.columns[i]),fontsize=30)
            plt.plot(y_predict_[:,i],color='blue')
            plt.plot(in_sample[:,i],color='red')
            plt.legend(['Predicted','Target'])
            plt.xlabel('Time(h)',fontsize=15)
            plt.ylabel('Value',fontsize=15)
        plt.savefig('results/{}/In_sample{}.png'.format(filepath,n_attempt))
        #plt.show()
        plt.close()
        '''
        #Plot predicted results and returns error metrics
        if plot_image:
            errors = plot_predict(lim=lim,yt_totest=yt_totest,num_series=self.num_series,data=data,out_sample=out_sample,trends=[],ndata=ndata,filename='{}/Outsample {}'.format(filepath,n_attempt), fig_axis=fig_axis)
        else:
            errors = None
        return yt_totest, errors






    def predict_pattern(self, initial_values, lags_used = [], ndata=[''], data=[], out_sample=[],h_prev=0,n_attempt=0,not_used_lag = False, filepath='',lim=0, defuzz_method='cog',fig_axis=[3,2], n_patterns=0, list_rules=None, wd_given=True):
        '''
        Function for pattern prediction. 
        \n Important features: 
        \n - n_pattern: Number of seasonal patterns.
        \n - list_rules: list of rules for each pattern.
        '''

        #preprocess_data = Preprocess(data,h_prev=h_prev,num_series=num_series)
        
        #in_sample, out_sample = preprocess_data.split_data()
        
        #yt, yp, yp_lagged = preprocess_data.delay_input(in_sample = in_sample, lag = lag)


        defuzz = Defuzzification(self.mf_params,self.num_series)

        yp_totest = initial_values
        yt_totest = np.zeros((h_prev,self.num_series))

        for h_p in range(h_prev):
            print('='*89)
            #Select which ruleset use now.

            rem = h_p % 168

            k = rem // 24

            print(f'Debug only, rem = {rem} and k = {k}')
            print('='*89)
            ensemble_antecedents = list_rules[k].rules
            ensemble_rules = list_rules[k].complete_rules

            #If weight matrix is not given, just fill with ones.
            if not wd_given:
                wd_ = np.ones((ensemble_rules.shape[0], self.num_groups, ensemble_rules.shape[1]))
            else:
                wd_ = list_rules[k].wd_



            print(f'Shape of ensemble rules is {ensemble_rules.shape}')

            mX_values_in = np.zeros((1,self.mf_params.shape[0],yp_totest.shape[1]))
            antecedents_activated = []
            it = 0
            for i in range(self.num_series):
                mf_params = self.mf_params[:,i]
                for j in range(self.lag):

                    mX, _ = self.Fuzzify.fuzzify(np.array([yp_totest[0,i*self.lag+j]]),mf_params,num_groups=self.num_groups)
                    mX_values_in[:,:,i*self.lag+j] = mX


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
                            antecedents_activated.append((i*self.lag+j,idx_nonzero[k]))

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
            for i in range(self.num_series):
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
            y_pred = defuzz.run(self.defuzz_method,w_todefuzz,show=False)
            
            #Store predicted value into yt_totest.
            yt_totest[h_p,:] = y_pred
            
            #Last step, we use the predicted output to compose input data.
            y_temp = np.zeros(yp_totest.shape)
            assert y_temp.shape == yp_totest.shape
            y_temp[0,1:] = yp_totest[0,0:yp_totest.shape[1]-1]
            for ii in range(self.num_series):
                #print(yp_totest[0,ii*lag])
                #print(y_pred[0][ii])
                #yp_totest[0,ii*lag] = y_pred[0][ii]
                y_temp[0,ii*self.lag] = y_pred[0][ii]
                #print(yp_totest[0,yp_totest.shape[1]-1])
            yp_totest = y_temp

        k = 1
        for i in range(self.num_series):
            plt.subplot(fig_axis[0],fig_axis[1],k)
            plt.title('Serie {}'.format(ndata.columns[i]),fontsize=30)
            plt.plot(yt_totest[:,i],color='blue')
            plt.plot(out_sample[:,i],color='red')
            plt.legend(['Predicted','Target'])
            plt.xlabel('Time(h)',fontsize=15)
            plt.ylabel('Value',fontsize=15)
            k += 1
    


        errors = plot_predict(lim=500,yt_totest=yt_totest,num_series=self.num_series,data=data,out_sample=out_sample,trends=[],ndata=ndata,filename='{}/Outsample {}'.format(filepath,n_attempt), fig_axis=fig_axis)
        print('Finished plot predict pattern')
        return errors

    
    def predict_batch(self,data, initial_values = [], lags_used = [], ndata=[''], in_sample=[], out_sample=[], agg_training=None,h_prev=0,n_attempt=0,wd_=[],ensemble_antecedents=[],ensemble_rules=[],not_used_lag = False, filepath='',lim=0, fig_axis=[3,2], show = False):
        '''
        Data: predicted values (horizon, serie, batch)
        Input: initial data for prediction
        '''
        results = np.zeros(data.shape)
        for i in range(data.shape[2]):
            yt_totest, _ = self.predict(initial_values[:,:,i], lags_used = lags_used, ndata=ndata, in_sample=[], out_sample=data[:,:,i], h_prev=h_prev,n_attempt=0,wd_=wd_,ensemble_antecedents=ensemble_antecedents,ensemble_rules=ensemble_rules,not_used_lag = False, filepath=filepath,lim=lim, fig_axis=fig_axis, show = show)
            results[:,:,i] = deepcopy(yt_totest)

