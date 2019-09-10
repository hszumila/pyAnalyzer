from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import cm
import numpy as np
from numpy import polyfit
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import gaussian_kde as kde
import inspect
from matplotlib import colors
from matplotlib.colors import Normalize
import pylab
import sys
from scipy.odr import *
import math
from matplotlib.backends.backend_pdf import PdfPages
import header

##################################
# Read in some data file.
# Save everything into arrays.
# Make plots. 
# Success. See pdf.
##################################

# Input data file:
header.dataToArray("data/inputFile2.txt")
# Output pdf plots:
#with PdfPages(r'C:\Users\hszumila\Desktop\Charts.pdf') as export_pdf:
with PdfPages(r'Charts.pdf') as export_pdf:

	###########################################################################
	# Linear fit to data with x,y errors (uses orthogonal distance regression)
	###########################################################################
	# This plots the points
	plt.errorbar(header.val_x, header.val_y, xerr=header.err_x, yerr=header.err_y, fmt='b^', markersize=5,alpha=0.25)
	plt.xlabel('x$_{value}$')
	plt.ylabel('y$_{value}$')
	plt.title('basic plot with x,y errors')

	# This defines the fit function
	xx = header.val_x
	def f(B, xx):
    		return B[0]*xx+B[1]
	linear = Model(f)
	mydata = RealData(xx, header.val_y, sx=header.err_x, sy=header.err_y)
	myodr = ODR(mydata, linear, beta0=[1., 0.])
	myoutput = myodr.run()
	myoutput.pprint()

	# Here we get the values from the fit function so we can plot it
	x1=[]
	y1=[]
	for ii in header.val_x:
    		x1.append(ii)
    		y1.append(myoutput.beta[1] + myoutput.beta[0] * ii)
	plt.plot(x1, y1, 'r--', linewidth=2)
	plt.text(10,40, 'y$_{value}$ = %.3f x$_{value}$ + %.3f' %(myoutput.beta[0],myoutput.beta[1]),color='red')
	plt.text(15,35, 'R$^{2}$ = %.3f' %(myoutput.res_var),color='red')
	plt.axis([0, 150, 0, 55])
	export_pdf.savefig()
	plt.close()

	######################################################################
	# Linear fit to data with y errors only
	######################################################################
	plt.errorbar(header.val_x, header.val_y, xerr=0, yerr=header.err_y, fmt='b^',markersize=5,alpha=0.25)	
	p,V = np.polyfit(header.val_x, header.val_y, deg=1, w=header.err_y, cov=True, full=False)
	x=[]
	y=[]
	x_extra = []
	y_extra = []
	weights = []
	ll=0
	for ii in header.val_x:
    		x.append(ii)
    		y.append(p[1] + p[0] * ii)
    		weights.append(1.0/header.err_y[ll])
    		ll+=1
	plt.plot(x, y, 'r-')
	x_extra = x + x[-1:]
	y_extra = header.val_y + header.val_y[-1:]
	weights.append(sys.float_info.epsilon)
	fit_extra, cov_extra = np.polyfit(x_extra, y_extra, 1, w=weights, cov=True)
	plt.text(10,40, 'y$_{value}$ = A x$_{value}$ + B',color='red')
	plt.text(15,35, 'A, slope: %.3f +/- %.3f' %(fit_extra[0],np.sqrt(cov_extra[0][0])),color='red')
	plt.text(15,30, 'B, intercept: %.3f +/- %.3f' %(fit_extra[1],np.sqrt(cov_extra[1][1])),color='red')
	plt.xlabel('x$_{value}$')
	plt.ylabel('y$_{value}$')
	plt.title('basic plot with y errors')
	plt.axis([0, 150, 0, 55])
	export_pdf.savefig()
	plt.close()

	######################################################################
	# 1D Histogram data with Gaussian fit
	######################################################################
	# Best fit of the data
	(mu, sigma) = norm.fit(header.distr)

	# Histogram the data
	n, bins, patches = plt.hist(header.distr, 30, density=1, facecolor='blue', alpha=0.75)

	# Add best fit
	y = norm.pdf(bins, mu, sigma)
	l = plt.plot(bins, y, 'r--', linewidth=2)

	# Plot it all
	plt.xlabel('data')
	plt.title('Histogram fit: $\mu=%.3f, \sigma=%.3f$' %(mu,sigma))
	plt.grid(True)
	export_pdf.savefig()
	plt.close()

	######################################################################
	# 2D Histogram data
	######################################################################
	data2d=[header.distr,header.distr2]
	densObj = kde(data2d)
	colours = header.makeColours(densObj.evaluate(data2d))
	plt.scatter( data2d[0], data2d[1], color=colours )
	plt.xlabel('distribution 1')
	plt.ylabel('distribution 2')
	#plt.show()
	export_pdf.savefig()
	plt.close()