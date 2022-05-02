
import numpy as np
import pandas as pd

def kalman(rawdata,tol=0.0025,sigR=1e-3,keepOriginal=True):
    '''
    This function fills out the missing values using the Low Kalman Smoothing technique.

    Args:
    rawdata: Numpy array with numeric values representing the whole dataframe.
    tol: Tolerance.
    sigR: Sigma.
    keepOriginal: Set to True if you want to return the full dataframe (not only the inferred missings).

    Returns:
    This function returns a pandas Dataframe with the transformed data.
    '''

    #selecting missing
    X = rawdata[~np.isnan(rawdata).any(axis=1)]
    
    m = np.mean(X,axis=0)

    print ('Computing SVD...')

    U, S, V = np.linalg.svd(X - m)
    
    print ('done')

    d = np.nonzero(np.cumsum(S)/np.sum(S)>(1-tol))[0][0]

    Q = np.dot(np.dot(V[0:d,:],np.diag(np.std(np.diff(X,axis=0),axis=0))),V[0:d,:].T)
    
    print ('Forward Pass')
    state = []
    state_pred = []
    cov_pred = []
    cov = []
    cov.insert(0,1e12*np.eye(d))
    state.insert(0,np.random.normal(0.0,1.0,d))
    cov_pred.insert(0,1e12*np.eye(d))
    state_pred.insert(0,np.random.normal(0.0,1.0,d))
    for i in range(1,rawdata.shape[0]+1):

        z =  rawdata[i-1,~np.isnan(rawdata[i-1,:])]
        H = np.diag(~np.isnan(rawdata[i-1,:]))
        H = H[~np.all(H == 0, axis=1)]
        Ht = np.dot(H,V[0:d,:].T)

        R = sigR*np.eye(H.shape[0])

        state_pred.insert(i,state[i-1])
        cov_pred.insert(i,cov[i-1] + Q)

        K = np.dot(np.dot(cov_pred[i],Ht.T),np.linalg.inv(np.dot(np.dot(Ht,cov_pred[i]),Ht.T) + R))
    
        state.insert(i,state_pred[i] + np.dot(K,(z - (np.dot(Ht,state_pred[i])+np.dot(H,m)))))
        cov.insert(i,np.dot(np.eye(d) - np.dot(K,Ht),cov_pred[i]))

    print ('Backward Pass')
    y = np.zeros(rawdata.shape)
    y[-1,:] = np.dot(V[0:d,:].T,state[-1]) + m
    for i in range(len(state)-2,0,-1):
        state[i] =  state[i] + np.dot(np.dot(cov[i],np.linalg.inv(cov_pred[i])),(state[i+1] - state_pred[i+1]))
        cov[i] =  cov[i] + np.dot(np.dot(np.dot(cov[i],np.linalg.inv(cov_pred[i])),(cov[i+1] - cov_pred[i+1])),cov[i])

        y[i-1,:] = np.dot(V[0:d,:].T,state[i]) + m

    if (keepOriginal):
        y[~np.isnan(rawdata)] = rawdata[~np.isnan(rawdata)]

    return y

def complete_data(df):
    '''
    This returns the complete data in the dataframe format.

    Args:
    df: Pandas DataFrame with missing values.

    Returns:
    Pandas DataFrame with full data.
    '''
    x=df.loc[:, df.columns != 'File']
    rawdata = np.array(x).astype('float')
    y = kalman(rawdata,0.0025,1e-3)
    df_Y= pd.DataFrame(y, columns=x.columns)
    df_Y["File"]=df.File
    return(df_Y)


