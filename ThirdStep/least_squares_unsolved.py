import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess(filename: str,Npoints=None):
    """
        This function just imports the data,
        tosses away the metadata, and appends
        a column of ones to the data array 
        for reasons discussed in the pdf
    """
    df_raw = pd.read_csv(filename)

    # a note about this:
    #   lambda is a keyword in python that is used to define a function 
    #   inline. This declaration f = lambda x: min(x,1) is the same as 
    #   def f(x: int):
    #       return min(x,1)

    # a second note:
    #   the word 'map' is generally used to indicate that some operation
    #   will be applied elementwise to every part of the object in 
    #   question. In this case, we are applying f to every label in 
    #   the training data. This is because we just want to be able 
    #   to tell if a digit is zero or some other number.
    f = lambda x: min(x,1)
    df_raw['label'] = df_raw['label'].map(f)
    
    # returns separately labels and pixels. As we've seen before, this
    # is just to make our application of least squares cleaner. We dont
    # need to flatten things because we're not currently planning to plot 
    # any of these. We leave the reshaping to another function anyway.
    data_array = np.array(df_raw) 
    labels = data_array[:,0]
    pixels = data_array[:,1:]

    # if your code is running too slow, you can pass in Npoints as a 
    # finite number, but it runs in about 1s for me with all points 
    if Npoints is not None:
        labels = data_array[0:Npoints,0]
        pixels = data_array[0:Npoints,1:]

    # this just adds a column of ones to the end of the numpy array
    # and it does it in a way that doesn't require the user to pass
    # in the size of the array, which is a neat trick
    pixels = np.append(pixels,np.ones((pixels.shape[0],1)),axis=1)

    return labels, pixels

def least_squares(targets, inputs):
    """ 
        This function needs to take the values given by 
        prepocess defined above, and it needs to return 
        the vector (as a numpy array) beta from equation (13)
        in the pdf. I think from my pdf, X = inputs Transpose
        so you'll have to keep that in mind. The program can
        be written in a single line

        hint: the matrix that i took the ''inverse'' of in  
        the pdf isnt actually invertible so use 
        numpy.linalg.pinv instead of numpy.linalg.inv
        to compute the inverse if need be
    """
    beta = np.zeros(inputs.shape[1])
    return beta 

def compare_ls(targets, inputs):
    """
        This function computes the difference between
        your solution's version of least squares regression
        and the build in function used by numpy. 
        You should get very close, like 10e-18 if you're using
        all the data points
    """
    beta_h = least_squares(targets,inputs)
    beta_true = np.linalg.lstsq(inputs, targets,rcond = None)[0]
    return (beta_h - beta_true) @ (beta_h - beta_true)

if __name__ == '__main__':
    # this will just import data and then compare your
    # implementation of least squares to the one that exists 
    # in numpy.linalg 
    labels, pixels = preprocess('../data/train.csv')
    print(compare_ls(labels,pixels))
