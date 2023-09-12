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


def predict(features, beta) -> int:
    """
        It's pretty standard to refer to the data of a single point
        as ``features'' for the record. One you have trained your 
        beta vector, you have a very straightforward way to cheaply
        make predictions. The expensive thing is coming up with the
        right beta in the first place. After you come up with a good 
        beta, you can just save it and use it when you please
    """
    # this function claims that zero cannot exist.
    # rewrite it to do something more sophistocated
    return 1 

def test_predictions(targets, inputs, beta):
    """
    I wrote this in basically the dumbest possible way for the sake of readability. 
    A more streamlined way to do this is, e.g.

    predictions = inputs @ beta 
    return (targets - predictions) @ (targets - predictions)

    which does exactly the same thing, but may be harder to understand. 
    """
    incorrect = 0
    for data_point in range(inputs.shape[0]):
        # ask least squares to guess if a number is a zero or something else
        predicted_outcome = predict(inputs[data_point], beta)
        actual_outcome = targets[data_point]

        # if the actual answer and the predicted answer arent the same, lsr made a mistake
        if predicted_outcome - actual_outcome != 0: 
            incorrect += 1
    return incorrect




if __name__ == '__main__':
    labels, pixels = preprocess('../data/train.csv')
    print(f'The 2--norm difference between homebrewed least squares and store--bought is {compare_ls(labels,pixels)}')
    beta = least_squares(labels, pixels)
    num_mistakes = test_predictions(labels,pixels,beta)
    print(f'number of mistakes was {num_mistakes}')


