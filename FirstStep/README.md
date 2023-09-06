# Step 1: Setting up a development Evironment

It is necessary to be flexible when installing packages and running scripts. 
My personal workflow involves calling python scripts from the command line, and using 
a text editor and a terminal. This is what I would advocate for, since this is the set of
tools that I am most comfortable helping with, should some issue arise. 

## Version control
 
For our version control needs, let's use `git` and github. 
Install it using [these](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) instructions. 
If you're using a mac, you can use [these](https://git-scm.com/download/mac) instructions along with [these instrucitons for installing homebrew](https://brew.sh/). 

Confirm that your install works by cloning this repository using the code 
`git clone https://amo004/MLPlayground`.
This command pulls a github repository into your local filesystem so that you can manipulate the code locally without having to manually download/upload changes. 
When you make changes to a set of files monitored by git, the changes are tracked by git. 
The purpose of git is to make it straightforward to revert or track changes that you make, so that you can work without being 
concerned about causing damage to your own work or to the work of others. 

When you make some progress and decide that you would like to save your work, the way to do this is to stage a **commit**. 
Try making a subdirectory in this directory called `MyFirstDirectory`. 
You can do this in the command line with the bash command 
`mkdir MyFirstDirectory`.

If you want to save this directory, the way to do this is to tell `git` that you want it to track `MyFirstDirectory`, and you want to save that instruction. 
You can accomplish this by using the commands
`git add MyFirstDirectory
git commit -m "I made my first directory"`

The string after the flag above is called a **commit message** and it serves to track your intentions with a commit. This comment is saved for future reference, so that you can remember the purposes of various commits. 

When we both make changes to the repository, it can happen that we have differing versions of a file or directory, and this is handled by making a **merge**. For now, we'll postpone talking about this until we start working on the same files. 
After you have saved your progress, you can ''upload'' your changes by using the command 
`git push`, which will require your github credentials. 

## Python and packages

I'm using Python 3.8. You can check your python version with the command 
`python --version` in the command line. It would be good to make sure we're using similar versions of python. 
I manage my python packages using a tool called `pip`. If you would like to use this tool but don't already use it, you can follow [these](https://docs.brew.sh/Homebrew-and-Python) instrucitons to install it. 

After you have this worked out, it is very straightforward to install the packages we'll need. 
You should install `numpy`, `matplotlib`, and `tensorflow`. The last of these we won't actually use for a while. 
Using pip you can install these packages with the command 
`pip install $PACKAGE_NAME`.

After you have installed these packages, try to run the script `hello_ml.py` that I've included in this repository. If you see the message `Hey, it worked!`, then your installs are working correctly
