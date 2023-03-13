import numpy as np
from sklearn.linear_model import LinearRegression

class Preprocess():
    '''
    Preprocessing class for e-autoMFIS. 
    Preprocess has two main focus: apply some preprocess methods, such as diff series and detrend; and define lagged inputs for the model.
    Available methods:
    - diff_series
    - detrend_series
    - split_data
    - delay_input
    '''

    def __init__(self, data, h_prev = 0, num_series = 0):
        self.data = data
        self.h_prev = h_prev
        self.num_series = num_series


    def diff_series(self):
        '''
        Classical series differentiation.
        OUTPUT: In-sample data (in_sample), Out-sample data (out_sample)
        '''
        diff_data = self.data[1:,:] - self.data[0:self.data.shape[0]-1,:]
        in_sample = diff_data[:diff_data.shape[0]-self.h_prev,:]
        out_sample = diff_data[diff_data.shape[0]-self.h_prev:,:]

        return in_sample, out_sample

    def detrend_series(self):
        '''
        Detrend method using Linear Regression.
        OUTPUT: In-sample data (in_sample), Out-sample (out_sample) data and trends of each serie (trends)
        '''
            
        detrended = np.zeros((self.data.shape))
        trends = np.zeros((self.data.shape))

        for i in range(self.data.shape[1]):
            x_detrend = [k for k in range(0, self.data.shape[0])]
            x_detrend = np.reshape(x_detrend, (len(x_detrend), 1))
            y = self.data[:,i]
            model = LinearRegression()
            model.fit(x_detrend, y)
            # calculate trend
            trend = model.predict(x_detrend)
            trends[:,i] = [trend[k1] for k1 in range(0,len(x_detrend))]

            # detrend
            detrended[:,i] = [y[k2]-trend[k2] for k2 in range(0, len(x_detrend))]
            
            in_sample = detrended[:detrended.shape[0]-self.h_prev,:]
            out_sample = detrended[detrended.shape[0]-self.h_prev:,:]

        return in_sample, out_sample, trends

    def split_data(self):
        '''
        Split data into in-sample data and out-sample data
        OUTPUT: In-sample data (in_sample) and Out-sample data (out_sample)
        '''
        in_sample = self.data[:self.data.shape[0]-self.h_prev,:]
        out_sample = self.data[self.data.shape[0]-self.h_prev:,:]

        return in_sample, out_sample

    def delay_input(self,in_sample=None, lag = 0):
        '''
        Prepare data for multivariate time series problem, creating delayed inputs. If in-sample data is not given, delay_input use
        the entire data instead.
        INPUT: in_sample (optional)
        OUTPUT: target (yt), non-delayed input (yp) and delayed-input (yp_lagged)
        '''
        if in_sample is not None:
            yt = np.zeros((in_sample.shape[0]-lag-1,self.num_series),dtype='float')
            yp = np.zeros((in_sample.shape[0]-lag-1,self.num_series), dtype='float')

            #Now delay inputs
            yp_lagged = np.zeros((in_sample.shape[0]-lag-1,self.num_series*lag),dtype='float')

            for i in range(self.num_series):
                yp[:,i] = in_sample[lag:in_sample.shape[0]-1,i]
                yt[:,i] = in_sample[lag+1:,i]
                for k in range(lag):
                    yp_lagged[:,i*lag+k] = in_sample[lag-k:in_sample.shape[0]-k-1,i]

        else:
            print('In-sample data not found. Using entire data instead')
            yt = np.zeros((self.data.shape[0]-self.h_prev-lag-1,self.num_series),dtype='float')

            #Todas as entradas defasadas 
            yp = np.zeros((self.data.shape[0]-self.h_prev-lag-1,self.num_series), dtype='float')
            yp_lagged = np.zeros((self.data.shape[0]-self.h_prev-lag-1,self.num_series*lag),dtype='float')

            for i in range(self.num_series):
                yp[:,i] = self.data[lag:self.data.shape[0]- self.h_prev - 1,i]
                yt[:,i] = self.data[lag+1:self.data.shape[0]- self.h_prev,i]
                for k in range(lag):
                    yp_lagged[:,i*lag+k] = self.data[lag-k:self.data.shape[0]- self.h_prev - k-1,i]

        return yt, yp, yp_lagged
