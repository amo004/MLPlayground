# Dataset Description
he data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top.

# What you should do 

Okay, so we have a bunch of data in `../data/train.csv` and we want to use some of that data to brush off our numpy, matplotlib, and pandas skills. 

1. Use `pandas` to import the data from `../data/train.csv`
2. Use `numpy` to rearrange the data therein into a shape which corresponds to the actual pixels of a square image. As a note `numpy.reshape` will be of use. 
3. Use `matplotlib.pyplot.imshow` to plot and save the image of the first row of `train.csv` as a `.png` image. 

# The contents of this directory
I have included my own solution to this problem in `make_pics.py`,  and I encourage you to try for a while before you look at what I did. I have also included the relevant image in `number.png`, so you can see what kind of image I'm looking for. 

