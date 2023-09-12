# Linear least squares regression  
So for this step of our journey, we will implement least squares regression. 
In the `least_squares.pdf` I have included what I think is a reasonably 
thorough exposition of the math of least squares regression

## The task
You should implement least squares regression in the relevant function in `least_squares_unsolved.py`. 
Neither of the python files are complete; I want to add more to the task since this is perhaps not enough to keep you busy. 

## Helpful advice
1. In the `pdf', I was sloppy about assuming a particular matrix was invertible... it's generally not.
2. It's usually silly to compute a matrix inverse, but this is sort of a naive first attempt at solving the problem anyway, so that's what we're going with. Use `numpy.linalg.pinv()` to compute the relevant inverse instead of `numpy.linalg.inv()`.
3. matrix multiplication in numpy is conveniently packaged into the function `numpy.matmul(A,B)`. I say ''convenient'' sarcatically. This is phenominally annoying to type all the time and it makes reading code harder. Fortunately, there is a build--in operator for this job, namely, `@`. for a pair of numpy arrays, `A`, and `B`, `A @ B == numpy.matmul(A,B)`
