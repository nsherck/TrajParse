import numpy as np
import scipy as sp
import scipy.stats
import math
from pymbar import timeseries
import stats_TrajParse as stats
from scipy.optimize import curve_fit
from TrajParse_Classes import Atom, Molecule

def CombineMolecularBondBondCorrelationFunctions(molecules):
	''' Finds the total bond-bond correlation function by averaging molecular ones. '''
	# Also calculate the average Magnitude of Bond-Bond Distances
	cnt = 0
	for molecule in molecules:
		if cnt == 0:
			temp_BBCorrelation = np.asarray(molecule.BondBondCorrelationFunction)
			temp_BBMagnitude   = molecule.moleculeBackboneBondVectorMagnitudesTimeandChainLengthAverage
			temp_BBVariance    = molecule.moleculeBackboneBondVectorMagnitudesTimeandChainVariance
			temp_BBMagnitudeChain   = np.asarray(molecule.moleculeBackboneBondVectorMagnitudesTimeAverage)
			temp_BBVarianceChain    = np.asarray(molecule.moleculeBackboneBondVectorMagnitudesTimeVariance)
		else:
			temp_BBCorrelation = np.add(temp_BBCorrelation, np.asarray(molecule.BondBondCorrelationFunction))
			temp_BBMagnitude   = np.add(temp_BBMagnitude, molecule.moleculeBackboneBondVectorMagnitudesTimeandChainLengthAverage)
			temp_BBVariance    = np.add(temp_BBVariance, molecule.moleculeBackboneBondVectorMagnitudesTimeandChainVariance)
			temp_BBMagnitudeChain = np.add(temp_BBMagnitudeChain, np.asarray(molecule.moleculeBackboneBondVectorMagnitudesTimeAverage))
			temp_BBVarianceChain  = np.add(temp_BBVarianceChain, np.asarray(molecule.moleculeBackboneBondVectorMagnitudesTimeVariance))
		cnt += 1
	TotalBBCorrelation = np.divide(temp_BBCorrelation,cnt)
	TotalBBMagnitude   = np.divide(temp_BBMagnitude,cnt)
	TotalBBVariance    = np.divide(temp_BBVariance,cnt)
	TotalBBMagChain    = np.divide(temp_BBMagnitudeChain,cnt)
	TotalBBVarChain    = np.divide(temp_BBVarianceChain,cnt)
	return TotalBBCorrelation, TotalBBMagnitude, TotalBBVariance, TotalBBMagChain, TotalBBVarChain

def ExpFunction(x,A,B):
	''' Correlation Function Fit Form '''
	return A*np.exp(-x/B)

def FitCurve(func, xdata, ydata):
	''' Fit data to a function form '''
	parameters_opt, parameter_covariance = curve_fit(func,xdata,ydata)
	return parameters_opt, parameter_covariance

def CalculateChainStatistics(molecules,DataFilename):
	''' Use stats.py to calculate per molecule statistically independent averages '''
	# N.S. TODO: Put this into the Molecule Class and save the data to each molecule object
	DoPymbarComparison = True # set to true to check against pymbar timeseries data
	mean_temp = []
	semcc_temp = []
	unbiasedvar_temp = []
	kappa_temp = []
	nsamples_temp = []
	nwarmup_temp = []
	pymbar_timeseries = []
	pymbar_statistics = [] # holds the [[mean,var,stderr,nsamples]] for each chain
	pymbar_data = [] # holds the decorrelated, equilibrium data
	f = open(DataFilename,"rw")
	for index,col in enumerate(range(len(molecules))): # enumerate starts at 0
		if DoPymbarComparison == True: # An alternative to the autoWarmupMSER by K. Delaney
			data = np.loadtxt(DataFilename,dtype='float',comments='#',usecols=(col+1))
			pymbar_timeseries.append(timeseries.detectEquilibration(data)) # outputs [t0, g, Neff_max] 
			t0 = pymbar_timeseries[index][0] # the equilibrium starting indices
			Data_equilibrium = data[t0:]
			print "length equilibrium data"
			print len(Data_equilibrium)
			g = pymbar_timeseries[index][1] # the statistical inefficiency, like correlation time
			indices = timeseries.subsampleCorrelatedData(Data_equilibrium, g=g) # get the indices of decorrelated data
			data = Data_equilibrium[indices] # Decorrelated Equilibrated data
			pymbar_data.append(data)
			data = np.asarray(data)
			np.savetxt("PyMbarData.txt",data)
			print "pymbar equilibrium data"
			print len(data)
			print t0
			pymbar_statistics.append([np.mean(data),np.var(data),np.sqrt(np.divide(np.var(data),len(data))),len(indices), len(Data_equilibrium), g]) # holds the [[mean,var,stderr,nsamples]] for each chain
		warmupdata, proddata, idx = stats.autoWarmupMSER(f,(col+1))
		nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor = stats.doStats(warmupdata,proddata)
		mean_temp.append(mean)
		semcc_temp.append(semcc)
		unbiasedvar_temp.append(unbiasedvar)
		kappa_temp.append(kappa)
		nsamples_temp.append(nsamples)
		nwarmup_temp.append(len(warmupdata))
	return mean_temp, semcc_temp, unbiasedvar_temp, kappa_temp, nsamples_temp, nwarmup_temp, pymbar_statistics, pymbar_data

def nint(x):
	for i,j in enumerate(x):
		x[i] = round(j)
	return x	
	
def WrapCoarseGrainCoordinates(Pos,box):
	''' Wraps coarse-grained segments through PBC '''
	# Code takes two arguments. One is the positions of the atoms, the second is the box dimensions on that step
	Pos_temp = []
	for index,i in enumerate(Pos):
		if (i/box[index]) < 0: # below the box
			Pos_temp.append((i+-1*box[index]*math.floor(i/box[index])))
		elif (i/box[index])> 1: # above the box 
			Pos_temp.append((i-box[index]*math.floor(i/box[index])))
		else:
			Pos_temp.append(i)
	return Pos_temp
	
def DoBootStrappingOnHistogram(data,hist,NumberBins,RangeMin,RangeMax):
	''' Generate sample data sets from histogram-ed data '''
	# The PDF is zero above(below) the highest(lowest) bin of the histogram defined by max(min) 
	#	of the original dataset.
	''' BootStrapping Options '''
	NormHistByMax = True
	BasicBoot = False # the better default, see Wikipedia BootStrapping
	PercentileBoot = True
	alpha = 0.05
	NumberBootStraps = 10000
	''' ********************* '''
	LenData = len(data) # generate bootstrap data sets with same number of samples
	hist_dist = scipy.stats.rv_histogram(hist) # for continuous data
	GenDataSets = []
	GenHistograms = []
	for i in range(NumberBootStraps): # Generated fictitious data sets from pdf
		temp      = hist_dist.rvs(size=LenData)
		GenDataSets.append(temp)
		tempHist  = np.histogram(temp,NumberBins,range=(RangeMin,RangeMax),density=True)
		if NormHistByMax == True:
			tempHist2 = np.divide(tempHist[0],np.max(tempHist[0])).tolist()
		else:
			tempHist2 = tempHist[0].tolist()
		GenHistograms.append(tempHist2)
	HistBins = tempHist[1] # Get the histogram bins (the same for all bins)
	GenHistogramArray = np.asarray(GenHistograms)
	HistAvg = np.mean(GenHistogramArray,axis=0).flatten()
	HistStdDev = np.std(GenHistogramArray,axis=0).flatten()
	HistStdErr0 = HistStdDev/np.sqrt(NumberBootStraps)
	HistUpperPercentile = np.percentile(GenHistogramArray,(100*(1-alpha/2)),axis=0).flatten()
	HistLowerPercentile = np.percentile(GenHistogramArray,(100*alpha/2),axis=0).flatten()
	HistStdErr1 = scipy.stats.sem(GenHistogramArray,axis=0).flatten()
	if PercentileBoot == True:
		tempPlus = HistUpperPercentile
		tempMinus = HistLowerPercentile
	if BasicBoot == True:
		tempPlus = np.subtract(2*HistAvg,HistLowerPercentile)
		tempMinus = np.subtract(2*HistAvg,HistUpperPercentile)
		
	#CIPlus = np.add(HistAvg,(Zscore95*HistStdErr0))
	#CIMinus = np.subtract(HistAvg,(Zscore95*HistStdErr0))
	
	CIPlus = tempPlus
	CIMinus = tempMinus
	
	#Overall averages and stderr
	GenDataSets = np.asarray(GenDataSets)
	HistAvgValue = np.mean(GenDataSets)
	HistAvgValueStdDev = np.std(GenDataSets)
	
	return HistAvg, CIPlus, CIMinus, alpha, NormHistByMax, HistAvgValue, HistAvgValueStdDev
	
def PickleObjects(filename,data2pickle, Debug_Pickling):
	''' Save the filename '''
	import pickle as pickle
	f = open(filename, 'wb')
	pickle.dump(data2pickle,f) # pickling the data
	f.close()
	
	if Debug_Pickling == 1:
		f_open = open(filename)
		import_from_pickle = pickle.load(f_open)
		if filename == 'atoms.pickled':
			print ("atom 1 ID:")
			print (import_from_pickle[0].atomID)
		elif filename == 'molecules.pickled':
			print ("molecule 1 ID:")
			print (import_from_pickle[0].moleculeID)
			
			