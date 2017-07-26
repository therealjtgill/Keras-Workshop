# Keras-Workshop
Keras Workshop for Tucson Python Meetup.
The focus of this workshop is to get you acquainted with Keras, so all of the Matplotlib (sorry, __[@hclent](https://github.com/hclent)__) and numpy/scipy stuff is taken care of for you.

# How to do dis
 - Read the first project, complete each of the tasks. Once you finish, you'll have implemented your own feed-forward neural network for binary classification. (yay!)
 - Once you've played with the first project to your heart's content, try the second project
 - The projects were written in Jupyter. If you don't have Jupyter, you can edit the *.py files, it's just not as much fun
 - The Jupyter notebooks have links to relevant information (Keras documentation, Wiki pages, memes, etc.)

## Install

* Meetup site: <https://www.meetup.com/Tucson-Python-Meetup/events/241824940/>
* Installation for Mac/Linux (This was done on macOS Sierra):

	```
	mkproject tensorflow
	workon tensorflow
	pip install -r requirements.txt
	echo "hooray!"
	```
	
## Demo

```
brew install python3
pip3 install matplotlib

mkproject -p python3 tensorflow
workon tensorflow
git clone https://github.com/therealjtgill/Keras-Workshop.git
cd Keras-Workshop/

# Install pip requirments
## via requirements
pip install -r requirements.txt

## via pip manually
pip install tensorflow
pip install numpy
pip install scipy
pip install keras
pip install matplotlib
pip install scikit-learn
pip install jupyter

# run notebook
jupyter notebook

# If you get the Mac error (missing backend blah blah blah), create this file:
$ cat ~/.matplotlib/matplotlibrc
backend: TkAgg
$

```

Mac Specifics:

* <https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python>