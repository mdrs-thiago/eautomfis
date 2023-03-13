import numpy as np
import matplotlib.pyplot as plt
from basicfuzzy import *

class Defuzzification():
    '''
    Defuzzification class for e-autoMFIS model.
    \n This class implements methods for Defuzzification. Available methods:
    \n - 'cog': Center of Gravity
    \n - 'mom': Mean of Maximum
    \n - 'height': Height method
    '''
    def __init__(self,mf_params,num_series):
        self.mf_params = mf_params
        self.num_series = num_series

    def run(self, name, agg_training, show=False):
        if name == 'cog':
            return self.defuzz_cog(self,agg_training,show) 
        if name == 'mom':
            return self.defuzz_mom(self,agg_training,show)
        if name == 'height':
            return self.defuzz_height(self,agg_training,show)
        else:
            print('Function for Defuzzification not found.')


    @staticmethod
    def defuzz_cog(self,agg_training,show=False):
        y_predict_ = np.zeros((agg_training.shape[0],self.num_series))
        for i in range(self.num_series):

            a = int(self.mf_params[-1,i] - self.mf_params[0,i])
            support_discourse = np.linspace(1.5*self.mf_params[0,i] - self.mf_params[1,i] ,2*self.mf_params[-1,i] - self.mf_params[-2,i],num=max(500,a))
            all_values = np.zeros((support_discourse.shape[0],self.mf_params.shape[0]))

            for j in range(self.mf_params.shape[0]):
                if j == 0:
                    k = 0
                    for val in support_discourse:
                        all_values[k,j] = trapmf(val,-1000*abs(self.mf_params[j,i]),-1000*abs(self.mf_params[j,i]),self.mf_params[j,i],self.mf_params[j+1,i])
                        k += 1
                    #print(all_values[:,j,i])

                elif j < self.mf_params.shape[0] - 1:
                    k = 0
                    for val in support_discourse:
                        all_values[k,j] = trimf(val,self.mf_params[j-1,i],self.mf_params[j,i],self.mf_params[j+1,i])
                        k += 1

                else:
                    k = 0
                    for val in support_discourse:
                        all_values[k,j] = trapmf(val,self.mf_params[j-1,i],self.mf_params[j,i],1000*abs(self.mf_params[j,i]),1000*abs(self.mf_params[j,i]))
                        k += 1

            for p in range(agg_training.shape[0]):
                p_in = np.ones(shape=all_values.shape) * agg_training[p,:,i]  

                out = np.minimum(all_values,p_in)
                outResponse = np.maximum.reduce(out,axis=1)

                y_predict = sum(np.multiply(support_discourse,outResponse))/(sum(outResponse)+0.00001)

                y_predict_[p,i] = y_predict
                
                if show:
                    if (p%100 < 10) and (i == 0):
                        #plt.figure(figsize=(16,9))
                        plt.plot(support_discourse,out)
                        plt.title('#{} on Serie {} predicted y_hat = {}'.format(p,i,y_predict_[p,i]))
                        plt.show()
                        plt.close()
            
        return y_predict_
        
    @staticmethod
    def defuzz_mom(self,agg_training,show=False):
        y_predict_ = np.zeros((agg_training.shape[0],self.num_series))
        for i in range(self.num_series):

            a = int(self.mf_params[-1,i] - self.mf_params[0,i])
            support_discourse = np.linspace(2*self.mf_params[0,i] - self.mf_params[1,i] ,2*self.mf_params[-1,i] - self.mf_params[-2,i],num=max(500,a))
            all_values = np.zeros((support_discourse.shape[0],self.mf_params.shape[0]))

            for j in range(self.mf_params.shape[0]):
                if j == 0:
                    k = 0
                    for val in support_discourse:
                        all_values[k,j] = trapmf(val,-1000*abs(self.mf_params[j,i]),-1000*abs(self.mf_params[j,i]),self.mf_params[j,i],self.mf_params[j+1,i])
                        k += 1
                    #print(all_values[:,j,i])

                elif j < self.mf_params.shape[0] - 1:
                    k = 0
                    for val in support_discourse:
                        all_values[k,j] = trimf(val,self.mf_params[j-1,i],self.mf_params[j,i],self.mf_params[j+1,i])
                        k += 1

                else:
                    k = 0
                    for val in support_discourse:
                        all_values[k,j] = trapmf(val,self.mf_params[j-1,i],self.mf_params[j,i],1000*abs(self.mf_params[j,i]),1000*abs(self.mf_params[j,i]))
                        k += 1

            for p in range(agg_training.shape[0]):
                p_in = np.ones(shape=all_values.shape) * agg_training[p,:,i]  

                out = np.minimum(all_values,p_in)

                outResponse = np.maximum.reduce(out,axis=1)

                max_index = max(outResponse)
                if (max_index > 0.0):
                    mom = [idd for idd,jj in enumerate(outResponse) if jj==max_index]
                    
                    y_predict = 0.5*(support_discourse[mom[0]]+support_discourse[mom[-1]])

                    y_predict_[p,i] = y_predict
                else:
                    y_predict_[p,i] = 0.0
                
                if show:
                    if (p%100 < 10) and (i == 0):
                        #plt.figure(figsize=(16,9))
                        plt.plot(support_discourse,outResponse)
                        plt.title('#{} on Serie {} predicted y_hat = {}'.format(p,i,y_predict_[p,i]))
                        plt.show()
                        plt.close()
            
        return y_predict_

    @staticmethod
    def defuzz_height(self,agg_training,show=False):
        y_predict_ = np.zeros((agg_training.shape[0],self.num_series))
        for i in range(self.num_series):

            a = int(self.mf_params[-1,i] - self.mf_params[0,i])
            support_discourse = np.linspace(2*self.mf_params[0,i] - self.mf_params[1,i] ,2*self.mf_params[-1,i] - self.mf_params[-2,i],num=max(500,a))
            all_values = np.zeros((support_discourse.shape[0],self.mf_params.shape[0]))
            #print('a')
            for j in range(self.mf_params.shape[0]):
                if j == 0:
                    k = 0
                    for val in support_discourse:
                        all_values[k,j] = trapmf(val,-1000*abs(self.mf_params[j,i]),-1000*abs(self.mf_params[j,i]),self.mf_params[j,i],self.mf_params[j+1,i])
                        k += 1
                    #print(all_values[:,j,i])

                elif j < self.mf_params.shape[0] - 1:
                    k = 0
                    for val in support_discourse:
                        all_values[k,j] = trimf(val,self.mf_params[j-1,i],self.mf_params[j,i],self.mf_params[j+1,i])
                        k += 1

                else:
                    k = 0
                    for val in support_discourse:
                        all_values[k,j] = trapmf(val,self.mf_params[j-1,i],self.mf_params[j,i],1000*abs(self.mf_params[j,i]),1000*abs(self.mf_params[j,i]))
                        k += 1
            #print('b')
            for p in range(agg_training.shape[0]):
                p_in = np.ones(shape=all_values.shape) * agg_training[p,:,i]  

                out = np.minimum(all_values,p_in)

                store_height = 0
                store_den = 0
                for n_set in range(out.shape[1]):
                    outResponse  =out[:,n_set]
                    max_value = max(outResponse)
                    if (max_value > 0.0):
                        mom = [idd for idd,jj in enumerate(outResponse) if jj==max_value]
                        
                        y_ = 0.5*(support_discourse[mom[0]]+support_discourse[mom[-1]])

                        store_height += y_*max_value
                        store_den += max_value
                    else:
                        pass
                if (store_height == 0):
                    y_predict_[p,i] = 0
                else:
                    y_predict_[p,i] = store_height/(store_den + np.finfo(float).eps)
                    #print(y_predict_[p,i])

                if show:
                    if (p%100 < 10) and (i == 0):
                        #plt.figure(figsize=(16,9))
                        plt.plot(support_discourse,out)
                        plt.title('#{} on Serie {} predicted y_hat = {}'.format(p,i,y_predict_[p,i]))
                        plt.show()
                        plt.close()
            
        return y_predict_


if __name__ == "__main__":

    mf_p = np.array([0.3, 0.5, 0.7, 1.0, 1.2])

    agregated_data = np.random.rand(5,5,1)

    defuzzify = Defuzzification(mf_params=mf_p, num_series = 1)

    defuzzify.run('cog',agregated_data,show=True)