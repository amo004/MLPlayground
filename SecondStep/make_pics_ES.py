import numpy as np
import pandas as pd
import matplotlib.pyplot

# Drawing of digit of interest
entry_row_number = 3

# Reads csv file as a pandas data frame
data = pd.read_csv('../data/train.csv')

# Obtains user-indicated row from the data frame
row = data.iloc[entry_row_number]

# Converts row from data frame to np array and removes digit identifier [1:]
image_vector = np.array((row.iloc[1:]))

# Reshapes the image pixel values into a square array (matrix)
image_matrix = np.reshape(image_vector, [28,28])

# Shows image
matplotlib.pyplot.imshow(image_matrix, cmap = 'gray')
matplotlib.pyplot.show()