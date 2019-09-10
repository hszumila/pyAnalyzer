import numpy as np
from scipy.stats import norm
from matplotlib.colors import Normalize
from matplotlib import cm

# define global stuff
val_x = []
val_y = []
err_x = []
err_y = []
distr = []
distr2 = []

# returns array of data
def dataToArray(inputFile):
	file = open(inputFile,"r")
	data = file.readlines()
	for line in data:
		value = line.strip().split(",") #assumes values separated by comma
		val_x.append(float(value[0]))
		val_y.append(float(value[1]))	
		err_x.append(float(value[2]))
		err_y.append(float(value[3]))
		distr.append(float(value[4]))
		distr2.append(float(value[5]))
	
def makeColours( vals ):
	colours = np.zeros( (len(vals),3) )
	norm = Normalize( vmin=vals.min(), vmax=vals.max() )

	#Can put any colormap you like here.
	colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]
	return colours