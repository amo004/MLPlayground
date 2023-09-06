import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def import_data(filename: str):
    # filename is a string, and we want to use it to import data from ../data/train.csv
    # this function is kind of a silly one to define because we're just wrapping a pandas function
    df = pd.read_csv(filename)
    return df

def get_pixel_array(df,ind: int):
    # given a row of the dataframe from the training file above, we want to plot a number using matplotlib
    # but in order to do this, we need to actually access the image data in a usable format. 

    # we have here accessed a row of the dataframe
    row = df.iloc[ind]

    # we convert a row of the dataframe into a numpy array, which will toss away the metadata
    r = np.array(row)

    # what does this line do, and why?
    r = r[1:]

    # now we have to unflatten the array
    r = np.reshape(r,(28,28)) 

    return r

def plot_image(r):
    plt.imshow(r,cmap="Greys")
    plt.savefig('number.png')
    plt.close()
    
def main():
    df = import_data('../data/train.csv')
    arr = get_pixel_array(df,1)
    plot_image(arr)
    
    return 0

if __name__ == "__main__":
    main()
