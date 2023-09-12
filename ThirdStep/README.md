# Linear least squares regression  
So for this step of our journey, we will implement least squares regression. 
In the `least_squares.pdf` I have included what I think is a reasonably 
thorough exposition of the math of least squares regression

## The task
You should implement least squares regression in the relevant function in `least_squares_unsolved.py`. 

There is also the question of how to test the effectiveness of our algorithm at making prediction. 
In the digit data set, there are 42000 entries, and roughly 10 percent of those are zeros. So, if we 
were asked to make a program that will guess whether or not a digit is zero, one strategy to consider would be 
to simply declare that no digit can be zero, and write a simple program with that function. In this case, if we tested on the training data, 
we would be correct about 90 percent of the time. In the context of machine learning, this is a pretty good success rate. 
Is least squares better than just cclaiming that nothing can be zero? 

Normally, the way that you answer this question would be somewhat involved. It is common to split the training data into sets, using some of the available data to test the model you create and not to train. For now, we'll bruch that under the rug and just test on our training data. In `least_squares_unsolved.py`, I have written a function which takes in a trained beta vector along with the data from a single point and declares that there is no such thing as a zero in our problem. You should rewrite that prediction function after reading section 1.2 of the pdf.

In my implementation, I found that only `~600` mistakes were made by my trained beta vector, which makes it significantly better than 90% accurate.  
Precisely, it's about 98 percent accurate, which is good enough to arouse suspicion (which will be the subject of discussion later).

## Helpful advice
1. In the `pdf', I was sloppy about assuming a particular matrix was invertible... it's generally not.
2. It's usually silly to compute a matrix inverse, but this is sort of a naive first attempt at solving the problem anyway, so that's what we're going with. Use `numpy.linalg.pinv()` to compute the relevant inverse instead of `numpy.linalg.inv()`.
3. matrix multiplication in numpy is conveniently packaged into the function `numpy.matmul(A,B)`. I say ''convenient'' sarcatically. This is phenominally annoying to type all the time and it makes reading code harder. Fortunately, there is a build--in operator for this job, namely, `@`. for a pair of numpy arrays, `A`, and `B`, `A @ B == numpy.matmul(A,B)`
