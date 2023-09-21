import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd


def partition(df,testDensity: float):
    # This is just a homebrewed implementation of a funciton which already exists in scikit-learn and other libs
    # takes a data frame and a randomly selects testDensity% of the data to be used as testing data 
    outputs = np.array(df['output'])
    inputs = np.array(df['input'])
    numDataPoints = inputs.shape[0]

    # it's generally good to sample training/validation data randomly because you can get unexpected 
    # behavior if your data is sorted in some unintuitive way or something
    samples = np.random.rand(numDataPoints)
    indecesTest = []
    indecesTrain = []
    for s in range(samples.shape[0]):
        if samples[s] <= testDensity:
            indecesTest.append(s)
        else:
            indecesTrain.append(s)
    
    outputsTest, outputsTrain = outputs[indecesTest], outputs[indecesTrain]
    inputsTest, inputsTrain= inputs[indecesTest], inputs[indecesTrain]

    return inputsTrain, outputsTrain, inputsTest, inputsTrain

def appendOnes(arr):
    # this only exists as a little tool to start our analysis
    ones = np.ones(arr.shape[0])
    return np.transpose(np.array([arr,ones]))

def preprocess(arr, nvals):
    output = [arr, arr**0]
    if nvals == []:
        return np.transpose(np.array(output))
    for n in nvals:
        output.append(arr**n)
    return np.transpose(np.array(output))



def main(fname):
    df = pd.read_csv(fname)
    xT, yT, xTT,yTT = partition(df,0.2)
    print(f'size of train= {xT.shape[0]}')
    print(f'size of test= {xTT.shape[0]}')

    onesAppT = appendOnes(xT)
    onesAppTT = appendOnes(xTT)
    beta = np.linalg.lstsq(onesAppT,yT)[0]

    plt.plot(xT,yT,label='data',linewidth=0,marker='x')
    plt.plot(xT, beta[0]*xT + beta[1]*np.ones(xT.shape[0]),markersize=0,label='linear least squars')
    plt.legend()
    plt.savefig('lls.png')
    plt.close()

    nvals = [2,3,4,5]
    # xPost = preprocess(xT,nvals)
    # betap = np.linalg.lstsq(xPost,yT)[0]
    # print(betap)
    plt.plot(xT,yT,label='data',linewidth=0,marker='x')
    a = np.argsort(xT)
    for n in range(5):
        xPost = preprocess(xT,nvals[0:n])
        betap = np.linalg.lstsq(xPost,yT)[0]
        plt.plot(xT[a], (xPost @ betap)[a],markersize=0,label = f'nmax = {n+1}')
    plt.legend()
    plt.savefig('nlls.png')
    plt.close()

if __name__ == "__main__":
    main('cooked.csv')

