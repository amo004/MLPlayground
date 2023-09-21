
# A detour into fit quality and preprocessing

We have implemented the simplest of prediction tools at this point, and now it's time to talk about some data practices that are commonplace. 
I mentioned in another module that it's common practice to split your data into training and testing to tune your model. In what follows, 
we will actually do this, with some data that I have cooked up for visualizability and which can be found in `cooked.csv`.

The setup is basically as follows: there is some relationship between input and output, and we want to ascertain that relationship approximately. 
In reality, I have taken 300 random numbers uniformally sampled from the interval $$ [0,2 \pi]$$ and I constructed the ''output'' field by using code along the lines of 
`
input = np.random.rand(300)
output = np.sin(300) + np.random.rand(300) * 0.1
`
and we are trying to see how close we can get to this relation. 
I have added some noise to make things a little more interesting. 

## Partitioning data

I have given you a single set of data with only a ''training'' set. In the real world, you harvest data from some source and then use it to 
build some model, but it's necessary to somehow validate that your data actually makes good predictions. You would prefer not to do this in a 
production setting because it's unnecessarily risky. 

I like splitting my data into 20 percent testing and 80 percent training. There are holy wars fought about how much of your data should be training
and how much should be testing; I am no great authority on this subject but people care about it. This is pretty simple to do, for what it's worth. 
All that you  need to do is import data from the included `.csv` file and split it such that some fraction of the data won't be used to train the model that you're working with. 

## Underfitting 

For now, it's fine to use the least squares implementation from whatever library you like. I'll use `numpy.linalg.lstsq()` for mine. Partition your data into training and testing, and then use least squares regression on the training data. Plot the least squares fit on top of the training data. 
You should get something like this: 

