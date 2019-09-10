# analyzer

Contains some generic code that can be used for analyzing raw data

The best installation of python for data analysis is Anaconda (heavy reliance on NumPy package). The examples below are tested with v3.7:
```
   https://www.anaconda.com/distribution/#download-section
```
To run python, you can search for `Anaconda` in the Windows menu. In the terminal window, navigate to the appropriate location for working and then type `python` to enter the environment:
```
   cd path/to/work/folder
   python
``` 
## Some basic examples:
* **hello.py** can be run in the python environment by typing: `import hello`  
* **fibo.py** contains two routines. First type `import fibo`, then you can execute the routines (for example) as `fibo.fib(500)`  
* **plotter.py** makes some plots using some dummy data. In the directory "data", I generated some random text data. This data is read in using the header file, "header.py". The data is plotted using the various features in "plotter.py". To run this program in the python environment, type `import plotter`. This will output all plots to a .pdf file called "Charts.pdf".   
