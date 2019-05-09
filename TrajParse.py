import numpy as np
import scipy as sp
import scipy.stats
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'errorbar.capsize': 8}) #makes endcaps of errorbars visible
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import subprocess as prcs
import mdtraj as md
import mdtraj.reporters
import time
from DataFileTemplates import ParseLAMMPSDataFile
from TrajParse_Classes import Atom, Molecule 
from TrajParse_Functions import CombineMolecularBondBondCorrelationFunctions, ExpFunction, FitCurve, CalculateChainStatistics, nint, WrapCoarseGrainCoordinates, DoBootStrappingOnHistogram

with open('Timing.LOG', 'w') as f:
	f.write('File with timing breakdowns. \n')
Overall_start_time = time.time()

with open('Print.LOG', 'w') as pLog:
	pLog.write('File recording the calculation progress. \n')
	
''' TEST ''' 

#traj_test = md.load_lammpstrj('dump.coords_u_s.dat', top='PreNVTProduction.pdb')
#traj_test = md.load_dcd('npt_production_output_unwrapped.dcd',top='initial.pdb')
#print traj_test.n_frames
#print traj_test.xyz[0]


''' Debugging flags '''
Debug = 0 			# Debug Flag. If "1" on, if "0" off. 
Debug_ParseData = 0 # Debug just the parse lmp.data file 
Debug_Pickling = 0  # Debug pickling saving and loading for atom and molecule objects
Debug_PersistenceLength = 0
Debug_CGing = 0

''' USER INPUTS ''' 
# The TrajFileFormat is likely not going to be used and is deprecated 
TrajFileFormat = "PDB" # either LAMMPS or PDB || Currently not used!!
ReadDataFile = 1 # read in LAMMPS data/topology 
LammpsInputFilename = "system_EO7.data" # even if importing from .dcd need to have a LAMMPS topology file
startStep = 0 # Start on this index 
Read_Pickle = False
Pickle_Objects = False  # Do not pickle objects (also will not pickle if Read_Pickle is True, i.e. do not overwrite)
ShowFigures = False
''' DCD Specific '''
Infiles = ['chain_0.dcd']#,'chain_1.dcd']#,'chain_2.dcd','chain_3.dcd','chain_4.dcd']#,'nvt_production_output_wrapped_chain_1.dcd','nvt_production_output_wrapped_chain_2.dcd']
topfilename = 'EO7.pdb'
atomIndices = 51
Read_DCD    = True
TimestepImportFrequency = 10
''' LAMMPS Specific '''
Read_LAMMPSTraj = False
''' Calculations to Perform '''
CalculateCOM = True
CalculateRg = True # must also Calculate COM to calculate Rg
CalculateRee = True
CalculateBondBondCorrelationFunction = True
DoBootStrapping = True

''' Coarse-graining '''
CoarseGrainTrajectory = True
# #BackboneAtoms; #BackboneAtoms2oneMonomer; MonomerType
CGMappingBackbone = [[3, 3, 1],[15, 3, 2],[3, 3, 1]] # for the first 72 atoms, take the three backbone atoms and map them to one!

#startStep EXAMPLE: if output every 200 timesteps with 1 fs timestep, 
#		then a value of 5000 is 1 nanosecond.
MoleculeType = 'polymer_linear'
WriteAAAtomPositions = False
WriteCGAtomPositions_LAMMPSTRJ = True
LammpsOutputScaleCoord = 10
WriteCGAtomPositions_XYZ = False
WrapCGCoordinates = False
CGPairsIntermolecular = [[3,3],[3,1]]
CGPairsIntramolecular = [[3,3]]
WriteCGAtomRDFPairs = False

''' User input for bond-bond correlation function '''
# Currently only works for picking then skipping. Not vice, versa.
# EXAMPLE: for PEO, COCCOCCOC if you just want the bond-bond correlation for one monomer
#       and you want to use just C-O bonds, then set "Number_Picked_Atoms" = 2 and "Number_Skipped_Atoms" = 1

PickSkip = False
Number_Picked_Atoms = 2 #2, set to -9999 to not use
Number_Skipped_Atoms = 1

# Alternative to ALL atoms and PickSkip 
EveryNthAtom = 3
RemoveFirstNatoms = [1] # removes the first atom from the list, put the list index into this list

''' Put in from other script to add in bonding from the input data file '''
ATOM_TYPE_IGNORES_MOLECULES = [84,85] # Set the atom types to ignore for molecules
ATOM_TYPE_IGNORES_BACKBONE = [20] # Set the atom types to ignore in the polymer backbone

''' ************************************************************************************************************* '''
''' *********************************** END OF USER INPUTS ****************************************************** '''
''' ************************************************************************************************************* '''

if Read_LAMMPSTraj == True:
	''' Find out molecular topology '''
	# fills in atom bonding
	if ReadDataFile == 1:
		[atoms, molecules] = ParseLAMMPSDataFile(LammpsInputFilename,Debug_ParseData, ATOM_TYPE_IGNORES_MOLECULES, ATOM_TYPE_IGNORES_BACKBONE)
	else: 
		pass
	'''TODO: Import LAMMPSTrajectory '''

	
elif Read_DCD == True:
	if ReadDataFile == 1: # Read in a LAMMPS Data File 
		[atoms, molecules] = ParseLAMMPSDataFile(LammpsInputFilename,Debug_ParseData, ATOM_TYPE_IGNORES_MOLECULES, ATOM_TYPE_IGNORES_BACKBONE)
	else: 
		pass
	
	aIndices = range(atomIndices)
	''' Load in a DCD Traj. file. '''
	# Requires a .pdb file (for topology)
	print ("Loading in a .DCD trajectory file.")
	traj = md.load(Infiles,top=topfilename,atom_indices=aIndices)
	print ("unit cell")
	print ("{}".format(traj.unitcell_lengths[0])) # List of the unit cell on each frame
	print ('number of frames:')
	print ("{}".format(traj.n_frames))
	print ('number of atoms')
	print ("{}".format(traj.n_atoms))
	# Slices the input trajectory information to shorten
	shortTraj = traj.xyz[::TimestepImportFrequency]
	shortBoxLengths = traj.unitcell_lengths[::TimestepImportFrequency]
	
	cnt =  0
	for frame in shortTraj:
		cnt +=1
		for index,atom in enumerate(atoms):
			atom.Pos.append(frame[index])
	
	for atom in atoms:
		atom.Box = shortBoxLengths
	
elif Read_Pickle == True:
	print ("Reading in pickled atom and molecule objects.")
	import pickle as pickle
	filename_atoms = "atoms.pickled"
	filename_molecules = "molecules.pickled"
	
	f_open_atoms = open(filename_atoms)
	f_open_molecules = open(filename_molecules)
	atoms = pickle.load(f_open_atoms)
	molecules = pickle.load(f_open_molecules)
		
''' *********************************************************************************** '''
''' ******************** Begin Calculations on the Trajectories *********************** '''
''' *********************************************************************************** '''

if CalculateCOM == True:
	print ("Calculating Molecular Center of Masses.")
	for index,Molecule in enumerate(molecules):
		start_time_MinImage = time.time()
		Molecule.MolcularMinimumImage(atoms)
		end_time_MinImage = time.time()
		start_time_COM = time.time()
		Molecule.CalculateCenterOfMass(atoms)
		end_time_COM = time.time()
	
	f = open("Timing.LOG", "a")
	f.write("Minimum image runtime: {}\n".format((end_time_MinImage-start_time_MinImage)))
	f.write("Center-of-mass runtime: {}\n".format((end_time_COM-start_time_COM)))
	f.close()	

	if Debug == 1:
		print "molecule IDs and Center of masses:"
		for molecule in molecules:	
			print "molecule {} begin and end atoms {}".format(molecule.moleculeID, moleculeBeginEndAtoms)
			print "molecule {} center of mass: {}".format(molecule.moleculeID, molecule.CenterOfMass)

if CalculateRg == True:
	if CalculateCOM == False:
		print "Trying to calculate Rg without COM Calculation!"
	print "Calculating Radius of Gyration."
	for Molecule in molecules:
		start_time_Rg = time.time()
		Molecule.CalculateRadiusOfGyration(atoms)
		end_time_Rg = time.time()
	
		f = open("Timing.LOG", "a")
		f.write("Radius-of-Gyration runtime: {}\n".format((end_time_Rg-start_time_Rg)))
		f.close()

	''' Calculate Rg Average Quantities '''	
	print "Calculating Rg Average Quantities."
	cnt = 0
	Rg_list = []
	header = []
	Rg_temp = 0.0
	RgNormalizing = len(molecules[0].RadiusOfGyration)
	print "Length of Rg Data: {}".format(RgNormalizing)
	header.append("Step")
	for Molecule in molecules:
		Rg_temp = sum(Molecule.RadiusOfGyration) + Rg_temp
		Rg_temp1 = np.array(Molecule.RadiusOfGyration)
		Rg_temp1 = np.transpose(Rg_temp1)
		Rg_list.extend(Molecule.RadiusOfGyration) # ALL Rg Data
		if cnt == 0:
			Rg = Rg_temp1
			header.append(" Molecule_{}".format(cnt))
		else:
			Rg = np.column_stack((Rg,Rg_temp1))
			header.append(" Molecule_{}".format(cnt))
		cnt += 1

if CalculateRee == True:
	print "Calculating End-to-End Vector."
	for Molecule in molecules:
		start_time_Ree = time.time()
		Molecule.CalculateEndtoEndVector(atoms)
		end_time_Ree = time.time()
	
	f = open("Timing.LOG", "a")
	f.write("End-to-end vector runtime: {}\n".format((end_time_Ree-start_time_Ree)))
	f.close()	

	''' Calculate Ree Average Quantities '''
	print "Calculating Ree Average Quantities."
	cnt = 0
	Ree_list = []
	header = []
	Ree_temp = 0.0
	ReeNormalizing = len(molecules[0].EndtoEndVector)
	print "Length of Ree data: {}".format(ReeNormalizing)
	header.append("Step")
	for Molecule in molecules:
		Ree_temp = sum(Molecule.EndtoEndVector) + Ree_temp
		Ree_temp1 = np.array(Molecule.EndtoEndVector)
		Ree_temp1 = np.transpose(Ree_temp1)
		Ree_list.extend(Molecule.EndtoEndVector)  # ALL Ree Data
		if cnt == 0:
			Ree= Ree_temp1
			header.append(" Molecule_{}".format(cnt))
		else:
			Ree = np.column_stack((Ree,Ree_temp1))
			header.append(" Molecule_{}".format(cnt))
		cnt += 1

''' Save and plot end-to-end vector histogram. '''
scale = 1 # Change the length scale (e.g. from Angstroms to nm's)

if CalculateRg == True:
	# Radius of gyration distribution
	number_Rg_hist_bins = 50
	RgMinimumHistBin = 0
	Rg_max = (np.divide(np.asarray(Rg_list),scale)).max() + 0.05*(np.divide(np.asarray(Rg_list),scale)).max()
	hist = np.histogram(np.divide(np.asarray(Rg_list),scale), number_Rg_hist_bins, range=(0,Rg_max),density=True)
	Rg_hist = hist[0]
	Rg_bins = hist[1]
	if DoBootStrapping == True:
		[HistAvg, CIPlus, CIMinus, alpha, NormHistByMax, HistAvgValueRg, HistAvgValueStdDevRg] = DoBootStrappingOnHistogram(np.divide(np.asarray(Rg_list),scale),hist,number_Rg_hist_bins,RgMinimumHistBin,Rg_max)
	plt.hist(np.asarray(Rg_list),bins=number_Rg_hist_bins,density=True, facecolor='blue',alpha=0.2,edgecolor='black', linewidth=1.2)
	plt.xlabel("distance [nm]")
	plt.ylabel("probability density")
	plt.title("Rg distribution")
	plt.savefig('RgDistribution.png', format='png', dpi=1200)
	if ShowFigures == True:
		plt.show()
	plt.close()
	
	# With CI intervals
	plt.hist(np.asarray(np.divide(Rg_list,scale)),bins=number_Rg_hist_bins,range=(0,Rg_max),density=True, facecolor='blue',alpha=0.2,edgecolor='black', linewidth=1.2)
	plt.plot(Rg_bins[0:number_Rg_hist_bins],HistAvg,'b')
	plt.fill_between(Rg_bins[0:number_Rg_hist_bins],CIMinus,CIPlus,alpha=0.25,facecolor='r')
	plt.xlabel("distance [nm]")
	plt.ylabel("probability density")
	plt.title("Rg Distribution: {}% Confidence Intervals".format((100*(1-alpha))))
	plt.savefig('RgDistribution_CI.png', format='png', dpi=1200)
	
	with open("Rg_Distribution.txt", 'w') as f:
		f.write("#	Bin_end  Prob.-density \n")
		for i in zip(Rg_bins[:-1], Rg_hist):
			f.write("{} {} \n".format(i[0], i[1]))
			
	with open("Rg_Distribution_CI.txt", 'w') as f:
		f.write("#	Bin_start HistAvg CIPlus CIMinus \n")
		for i in zip(Rg_bins[0:number_Rg_hist_bins], HistAvg, CIPlus, CIMinus):
			f.write("{}   {}   {}   {} \n".format(i[0], i[1], i[2], i[3]))

if CalculateRee == True:
	# End-to-end distance distribution
	number_Ree_hist_bins = 30
	TrimRee = False
	ReeCutoff = 1.5
	ReeMinimumHistBin = 0.0
	
	# To remove values from Ree
	if TrimRee == True:
		Ree_temp2 = []
		Rg_temp2 = []
		cnt = 0
		for i,ReeValue in enumerate(Ree_list):
			if ReeValue >= ReeCutoff:
				Ree_temp2.append(ReeValue)
				Rg_temp2.append(Rg_list[i])
			else:
				cnt +=1
		Ree_list = Ree_temp2
		Rg_list  = Rg_temp2
		print "Ree values removed below a cutoff of {} were: {}".format(ReeCutoff,cnt)
	
	Ree_max = (np.divide(np.asarray(Ree_list),scale)).max() + 0.05*(np.divide(np.asarray(Ree_list),scale)).max()
	hist = np.histogram(np.divide(np.asarray(Ree_list),scale), number_Ree_hist_bins, range=(ReeMinimumHistBin,Ree_max),density=True)
	Ree_hist = hist[0]
	Ree_bins = hist[1]
	if DoBootStrapping == True:
		[HistAvg, CIPlus, CIMinus, alpha, NormHistByMax, HistAvgValueRee, HistAvgValueStdDevRee] = DoBootStrappingOnHistogram(np.divide(np.asarray(Ree_list),scale),hist,number_Ree_hist_bins,ReeMinimumHistBin,Ree_max)
	plt.hist(np.asarray(Ree_list),bins=number_Ree_hist_bins,density=True, facecolor='blue',alpha=0.2,edgecolor='black', linewidth=1.2)
	plt.xlabel("distance [nm]")
	plt.ylabel("probability density")
	plt.title("Ree distribution")
	plt.savefig('ReeDistribution.png', format='png', dpi=1200)
	if ShowFigures == True:
		plt.show()
	plt.close()
	
	# With CI intervals
	plt.hist(np.asarray(np.divide(Ree_list,scale)),bins=number_Ree_hist_bins,range=(ReeMinimumHistBin,Ree_max),density=True, facecolor='blue',alpha=0.2,edgecolor='black', linewidth=1.2)
	if NormHistByMax == True:
		Ree_hist_temp = np.divide(Ree_hist,Ree_hist.max())
	else:
		Ree_hist_temp = Ree_hist
	plt.plot(Ree_bins[0:number_Ree_hist_bins],Ree_hist_temp,'b')
	plt.fill_between(Ree_bins[0:number_Ree_hist_bins],CIMinus,CIPlus,alpha=0.25,facecolor='r')
	plt.xlabel("distance [nm]")
	plt.ylabel("probability density")
	plt.title("Ree Distribution: {}% Confidence Intervals".format((100*(1-alpha))))
	plt.savefig('ReeDistribution_CI.png', format='png', dpi=1200)
	
	with open("Ree_Distribution.txt", 'w') as f:
		f.write("#	Bin_end  Prob.-density \n")
		for i in zip(Ree_bins[:-1], Ree_hist):
			f.write("{} {} \n".format(i[0], i[1]))
			
	with open("Ree_Distribution_CI.txt", 'w') as f:
		f.write("#	Bin_start HistAvg CIPlus CIMinus \n")
		for i in zip(Ree_bins[0:number_Ree_hist_bins], Ree_hist_temp, CIPlus, CIMinus):
			f.write("{}   {}   {}   {} \n".format(i[0], i[1], i[2], i[3]))

''' Save and plot 2D Rg & Ree heat map. '''
Override = False
if CalculateRg == True and CalculateRee == True and Override == False:
	# Plot 2D heat map of the Rg and end-end distance
	number_hist_bins_x = 100
	number_hist_bins_y = 100
	H, xedges, yedges = np.histogram2d(np.divide(np.asarray(Ree_list),scale),np.divide(np.asarray(Rg_list),scale),bins=[number_hist_bins_x,number_hist_bins_y], normed=True)
	np.savetxt("RgRee_HeatMap.txt",H,header="# nx, ny")
	np.savetxt("RgRee_HeatMap_xedges.txt",xedges)
	np.savetxt("RgRee_HeatMap_yedges.txt",yedges)
	
	H = H.T
	fig = plt.figure(figsize=(6,6))
	ax = fig.add_subplot(111, title='imshow: RgRee_HeatMap')
	plt.imshow(H,interpolation='nearest',origin='low',aspect='equal', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]) #plt.imshow
	plt.savefig('RgRee_HeatMap.png', format='png', dpi=1200)
	if ShowFigures == True:
		plt.show()
	plt.close()

	#ax = fig.add_subplot(133, title='NonUniformImage: interpolated', aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
	#im = plt.NonUniformImage(ax, interpolation='bilinear')
	#xcenters = (xedges[:-1] + xedges[1:]) / 2
	#ycenters = (yedges[:-1] + yedges[1:]) / 2
	#im.set_data(xcenters, ycenters, H)
	#ax.images.append(im)
	#plt.savefig('RgRee_HeatMap_NonUniForm.png', format='png', dpi=1200)
	#plt.show()
	#plt.close()

	# Add in a first column with the index for use with python stats.
	Rg_temp2 = []
	Ree_temp2 = []
	for index,value in enumerate(Rg.tolist()):
		val_temp = []
		val_temp.append(index)
		if isinstance(value, (list,)): # check if only one chain in the system
			for x in value:
				val_temp.append(x)
		else: 
			val_temp.append(value)
		Rg_temp2.append(val_temp)
	for index,value in enumerate(Ree.tolist()):
		val_temp = []
		val_temp.append(index)
		if isinstance(value, (list,)):
			for x in value:
				val_temp.append(x)
		else:
			val_temp.append(value)
		Ree_temp2.append(val_temp)
	Rg_temp2 = np.asarray(Rg_temp2)
	Ree_temp2 = np.asarray(Ree_temp2)
	
	np.savetxt("Rg_u.txt",Rg_temp2,header="{}".format("".join(header)))
	np.savetxt("Ree_u.txt",Ree_temp2,header="{}".format("".join(header)))


	''' Calculate chain Rg and Ree statistics '''
	print "Calculating per molecule averages."
	filenames = ["Rg_u.txt", "Ree_u.txt"] # put Ree_u.txt second so that alpha can be calculated
	for file in filenames:
		mean, semcc, unbiasedvar, kappa, nsamples, nwarmup, pymbar_statistics, pymbar_data = CalculateChainStatistics(molecules,file)
		mean_total = 0.
		variance_total = 0.
		stderr_total = 0.
		kappa_total = 0.
		nsamples_total = 0.
		for j in mean:
			mean_total = mean_total + float(j)
		mean_total = mean_total/len(mean)
		for index, j in enumerate(unbiasedvar):
			variance_total 	= variance_total + float(j)
			kappa_total 	= kappa_total + float(kappa[index])
			nsamples_total	= nsamples_total + float(nsamples[index])
		stderr_total = np.sqrt(variance_total/nsamples_total) # (sum of variances/sum of samples)**0.5
		kappa_total = kappa_total/len(kappa)
		variance_total = variance_total
		stderr_total_cc = stderr_total*np.sqrt(kappa_total) # correlation corrected!, using average corr. time
		kappa_stddev = np.std(np.asarray(kappa))
		kappa_stderr = np.sqrt(np.square(kappa_stddev)/len(kappa))
		if file == "Rg_u.txt":
			RgAvg = mean_total
			RgStdErr = stderr_total_cc
		if file == "Ree_u.txt":
			ReeAvg = mean_total
			ReeStdErr = stderr_total_cc
			# calculate Ree/Rg = alpha: 
			alpha = ReeAvg/RgAvg
			alphaVar = (1/RgAvg)**2*ReeStdErr**2 + (ReeAvg/RgAvg**2)**2*RgStdErr**2
			alphaStdDev = np.sqrt(alphaVar)
		
		''' Save statistics from Rg and Ree. '''
		with open("{}_stats.txt".format(file), 'w') as f:
			f.write("#		")
			for index,i in enumerate(range(len(molecules))):
				f.write(" molecule_{}	".format(i))
			f.write("\n")
			f.write(" mean   	   {} \n".format(mean))
			f.write(" unbiasstdev  {} \n".format((np.sqrt(np.asarray(unbiasedvar))).tolist()))
			f.write(" unbiasedvar  {} \n".format(unbiasedvar))
			f.write(" kappa  	   {} \n".format(kappa))
			f.write(" nsamples 	   {} \n".format(nsamples))
			f.write(" nwarmup      {} \n".format(nwarmup))
			f.write(" npos. atom1  {} \n".format(len(atoms[0].Pos)))
			if len(pymbar_statistics) >= 1:
				f.write(" \n")
				f.write(" Averages calculated from pymbar: \n")
				f.write(" \n")
				f.write(" mean_total     {} \n".format(pymbar_statistics[index][0])) 
				f.write(" stderr_total   {} \n".format(pymbar_statistics[index][2]))
				f.write(" variance_total {} \n".format(pymbar_statistics[index][1]))
				f.write(" nsamples       {} \n".format(pymbar_statistics[index][3]))
				f.write(" nequil. data   {} \n".format(pymbar_statistics[index][4]))
				f.write(" corr. time     {} \n".format(pymbar_statistics[index][5]))
				f.write(" \n")
			f.write(" \n")
			f.write(" MOLECULE TOTALS: \n")
			f.write(" 	If there is a single molecule, use (unbiasedvar/nsamples)**(0.5). \n")
			f.write(" \n")
			f.write(" mean_total     {} \n".format(mean_total)) 
			f.write(" stderr_total   {} \n".format(stderr_total_cc))
			f.write(" variance_total {} \n".format(variance_total))
			if file == "Ree_u.txt":
				f.write(" alpha| Ree/Rg  {} \n".format(alpha))
				f.write(" alpha variance {} \n".format(alphaVar))
				f.write(" alpha stddev   {} \n".format(alphaStdDev))
			f.write(" \n")
			if file == "Ree_u.txt" and DoBootStrapping == True:
				f.write(" Averages from bootstrapping: \n")
				f.write(" \n")
				f.write(" mean_total     {} \n".format(HistAvgValueRee))
				f.write(" StdDev_total   {} \n".format(HistAvgValueStdDevRee))
				f.write(" \n")
			elif file == "Ree_u.txt" and DoBootStrapping ==False:
				f.write(" Averages from bootstrapping: \n")
				f.write(" - DoBootStrapping == False \n") 
			f.write(" CORRELATION TIME: \n")
			f.write(" \n")
			f.write(" correlation time  {} \n".format(kappa_total)) 
			f.write(" standard dev.   	{} \n".format(kappa_stddev))
			f.write(" standard error   	{} \n".format(kappa_stderr))
	
	
	print "Average Rg:"
	RgAve = Rg_temp/(cnt*RgNormalizing)
	print RgAve
	print "Average Ree:"
	ReeAve = Ree_temp/(cnt*ReeNormalizing)
	print ReeAve


''' Write all-atom positions in x,y,z format '''
#FORMAT:
#	<number of atoms>
#	comment line
#	atom_symbol11 x-coord11 y-coord11 z-coord11
#	atom_symbol12 x-coord12 y-coord11 z-coord12
#	...
#	atom_symbol1n x-coord1n y-coord1n z-coord1n
#
#	atom_symbol21 x-coord21 y-coord21 z-coord21
#	atom_symbol22 x-coord22 y-coord21 z-coord22
#	...
#	atom_symbol2n x-coord2n y-coord2n z-coord2n
#	.
#	.
#	.
#END FORMAT
if WriteAAAtomPositions == True:
	print "Writing out atom Positions"
	with open('AA_AtomPos.xyz', 'w') as f:
		TotalTimeSteps = len(atoms[0].Pos)
		NumberAtoms = len(atoms)
		print "Number atomistic atoms: {}".format(NumberAtoms)
		TotalIterations = NumberAtoms*TotalTimeSteps
		cnt = 0
		for step in range(TotalTimeSteps):
			f.write("{0:10} \n".format(NumberAtoms))
			f.write("# comment line\n")
			for atom in atoms:
				if len(atom.Pos) < TotalTimeSteps:
					print "WARNING: atom {} doesn't have all the time steps!".format(atom.atomID)
					pass
				tempPos = atom.Pos[step]
				f.write(" {0:<4} {1:<10.5f} {2:<10.5f} {3:<10.5f} \n".format(atom.atomType,tempPos[0],tempPos[1],tempPos[2]))
				cnt += 1
			#progress = cnt/TotalIterations
			
		
if Debug == 1:
	with open("DEBUG_Atom0_ImageFlags.txt", 'w') as f:
		cnt = 0
		f.write("Step ix iy iz")
		for i in atoms[0].ImageFlags:
			f.write("{} {}".format(cnt,i))
			cnt += 1

if CalculateBondBondCorrelationFunction == True:
	''' Calculate the magnitude of the bonding vectors along the polymer backbone '''
	print "Calculating molecular backbone bonding vectors."
	for molecule in molecules: 
		molecule.CalculateMolecularBondVectors(atoms, Number_Picked_Atoms, Number_Skipped_Atoms, RemoveFirstNatoms, EveryNthAtom, PickSkip)
	
	if Debug == 1:
		print "Molecule 1 Backbone First Bond Vectors:"
		print molecules[0].moleculeBackboneBondVectors[0]
		print "Molecule 1 Backbone First Bond Vector Magnitudes"
		print molecules[0].moleculeBackboneBondVectorMagnitudes[0]
	
	''' Calculate molecule bond-bond correlation function. '''
	for molecule in molecules:
		molecule.CalculateBondBondCorrelationFunction(atoms)
	
	if Debug == 1:
		print "Molecule 1 Bond-Bond Correlation Function"
		print molecules[0].BondBondCorrelationFunction
		print "Molecule 2 Bond-Bond Correlation Function"
		print molecules[1].BondBondCorrelationFunction

	''' Calculate ensemble and trajectory averaged total Bond-Bond correlation function. '''
	print "Calculating persistence length."
	TotalBBCorrelation, TotalBBMagnitude, TotalBBVariance, TotalBBMagChain, TotalBBVarChain = CombineMolecularBondBondCorrelationFunctions(molecules)
	if Debug_PersistenceLength == 1:
		print "Total Bond-Bond Correlation Function:"
		print TotalBBCorrelation

	''' Extract persistence length. '''
	# Units of the persistence length are currently dimensionless
	maxX = len(TotalBBCorrelation) - 1
	xdata = np.linspace(0,maxX,(maxX+1))
	parameters_opt, parameter_covariance = FitCurve(ExpFunction, xdata, TotalBBCorrelation)
	perr = np.sqrt(np.diag(parameter_covariance))
	# Calculate the persistence length and stderr
	PersistenceVariance = parameters_opt[1]**2*TotalBBVariance + TotalBBMagnitude**2*perr[1]**2
	PersistenceStdErr   = np.sqrt(PersistenceVariance)
	PersistenceLength   = parameters_opt[1]*TotalBBMagnitude
	
	
	if Debug_PersistenceLength == 1:
		print "xdata"
		print xdata
		print "Optimal Parameters are:"
		print parameters_opt
		print "Parameter standard errors are:"
		print perr
	
	''' Save Total Bond-Bond Correlation to a file. '''
	with open("TotalBBCorrelation.txt",'w') as f:
		f.write("# Bond-Separation(A.U.) CorrFnx: parameter_A = {}+/-{}, parameter_B = {}+/-{}, PersistenceLength = {}+/-{} \n".format(parameters_opt[0],perr[0],parameters_opt[1],perr[1], PersistenceLength, PersistenceStdErr))
		for index,value in enumerate(TotalBBCorrelation.tolist()):
			f.write("{} {}\n".format(index,value))
	
	''' Save Total Bond-Bond Correlation to a file. '''
	with open("TotalBBMagChain.txt",'w') as f:
		f.write("# bond index  bondMag  bondMagVariance \n")
		for index,value in enumerate(zip(TotalBBMagChain.tolist(),TotalBBVarChain.tolist())):
			f.write("{}  {}  {}\n".format(index,value[0],value[1]))
	
	''' Plot the bond-bond correlation function. '''
	fig, ax1 = plt.subplots()
	ax1.plot(TotalBBCorrelation, label='data', linewidth=3)
	if len(parameters_opt) != "":
		ax1.plot(xdata,ExpFunction(xdata, *parameters_opt),'r--', label='fit A = {0:2.3f}, B = {1:2.3f}, PersistenceLength = {2:2.3f}'.format(parameters_opt[0],parameters_opt[1],PersistenceLength), linewidth=3)
	ax1.legend()
	#ax1.axhline(linewidth=6)
	#ax1.axvline(linewidth=6)
	ax1.tick_params(axis='both',direction='in',width=2,length=6)
	ax1.set_title("Bond-Bond Correlation Function")
	ax1.set_xlabel("bond separation")
	ax1.set_ylabel("bond-bond correlation")
	plt.savefig('BondBondCorrelation.png', format='png', dpi=1200)
	if ShowFigures == True:
		plt.show()
	plt.close()

if CoarseGrainTrajectory == True:
	''' Map all-atom to coarse-grain system '''
	print "Mapping the AA trajectory to a CG trajectory."
	start_time = time.time()
	cgatoms = [] # instantiate CGatoms
	for molecule in molecules:
		molecule.CGCalculateCenterOfMass(atoms, CGMappingBackbone, cgatoms, Debug_CGing)
	end_time = time.time()
	f = open("Timing.LOG", "a")
	f.write("Coarse-graining atomistic trajectory runtime: {}\n".format((end_time-start_time)))
	f.close()

	
	if WriteCGAtomPositions_XYZ == True:
		print "Writing out atom Positions"
		with open('CG_AtomPos.xyz', 'w') as f:
			TotalTimeSteps = len(cgatoms[0].Pos)
			NumberAtoms = len(cgatoms)
			print "Number coarse-grained atoms: {}".format(NumberAtoms)
			TotalIterations = NumberAtoms*TotalTimeSteps
			cnt = 0
			for timestep,step in enumerate(range(TotalTimeSteps)):
				box = cgatoms[0].Box[step]
				f.write("{0:10} \n".format(NumberAtoms))
				f.write("# comment line\n")
				#f.write('Lattice="{0:5.3f} 0.0 0.0 0.0 {1:5.3f} 0.0 0.0 0.0 {2:5.3f}" Properties="species:S:1:pos:R:3"\n'.format(box[0],box[1],box[2]))
				for atomID, atom in enumerate(cgatoms):
					if len(atom.Pos) < TotalTimeSteps:
						print "WARNING: atom {} doesn't have all the time steps!".format(atom.atomID)
						pass
					tempPos = atom.Pos[step]
					f.write(" {0:<4} {1:<10.5f} {2:<10.5f} {3:<10.5f} \n".format(atom.atomType,tempPos[0],tempPos[1],tempPos[2]))
					cnt += 1
		
	if WriteCGAtomPositions_LAMMPSTRJ == True:
		print "Writing out atom Positions"
		ScaleCoordsBy = LammpsOutputScaleCoord # Scale the output coordinates
		start_time = time.time()
		with open('CG_AtomPos.lammpstrj', 'w') as g:
			TotalTimeSteps = len(cgatoms[0].Pos)
			NumberAtoms = len(cgatoms)
			print "Number coarse-grained atoms: {}".format(NumberAtoms)
			TotalIterations = NumberAtoms*TotalTimeSteps
			cnt = 0
			for timestep,step in enumerate(range(TotalTimeSteps)):
				box = cgatoms[0].Box[step]
				g.write('ITEM: TIMESTEP\n')
				g.write('{0:<8}\n'.format(timestep))
				g.write('ITEM: NUMBER OF ATOMS\n')
				g.write('{0:<10}\n'.format(NumberAtoms))
				g.write('ITEM: BOX BOUNDS pp pp pp\n')
				g.write('0.0000000000000000e+00 {0:12.12e}\n'.format((box[0]*ScaleCoordsBy)))
				g.write('0.0000000000000000e+00 {0:12.12e}\n'.format((box[1]*ScaleCoordsBy)))
				g.write('0.0000000000000000e+00 {0:12.12e}\n'.format((box[2]*ScaleCoordsBy)))
				g.write('ITEM: ATOMS id mol type xu yu zu\n')
				#f.write('Lattice="{0:5.3f} 0.0 0.0 0.0 {1:5.3f} 0.0 0.0 0.0 {2:5.3f}" Properties="species:S:1:pos:R:3"\n'.format(box[0],box[1],box[2]))
				for atomID, atom in enumerate(cgatoms):
					if len(atom.Pos) < TotalTimeSteps:
						print "WARNING: atom {} doesn't have all the time steps!".format(atom.atomID)
						pass
					tempPos = atom.Pos[step]
					if WrapCGCoordinates == True: # Check if atom coordinate wrapping has been specified? 
						tempPos = WrapCoarseGrainCoordinates(tempPos,box)
					g.write("{0:<8} {1:<6} {2:<4} {3:<10.5f} {4:<10.5f} {5:<10.5f}\n".format(atomID,atom.atomMol,atom.atomType,(tempPos[0]*ScaleCoordsBy),(tempPos[1]*ScaleCoordsBy),(tempPos[2]*ScaleCoordsBy)))
					cnt += 1
		end_time = time.time()
		f = open("Timing.LOG", "a")
		f.write("WriteCGAtomPositions_LAMMPSTRJ runtime: {}\n".format((end_time-start_time)))
		f.close()
		
	
	if WriteCGAtomRDFPairs == True:
		start_time = time.time()
		for index,pair in enumerate(CGPairsIntermolecular):
			atomType1 = pair[0]
			atomType2 = pair[1]
			with open('IntermolecularPairs_between_{}_{}'.format(atomType1,atomType2),'w') as f:
				for atom1ID, atom1 in enumerate(cgatoms):
					for atom2ID, atom2 in enumerate(cgatoms):
						if atom1ID == atom2ID: # exclude if the same atom
							pass
						elif atom1.atomMol != atom2.atomMol and atom1.atomMol < atom2.atomMol: 
						# exclude if on same molecule and if atom1.molecule is larger than atom 2 (i.e. pair already counted)
							if atom1.atomType == atomType1 and atom2.atomType == atomType2:
								f.write('{} {}\n'.format((atom1ID),(atom2ID)))
							else:
								pass							
	end_time = time.time()
	f = open("Timing.LOG", "a")
	f.write("Writeout atom pairs list runtime: {}\n".format((end_time-start_time)))
	f.close()			
		 
if Read_Pickle == False and Pickle_Objects == True:
	''' Save system objects atoms and molecules to pickled files. '''
	print "Pickling the atom and molecule objects."
	import pickle as pickle
	filename_atoms = "atoms.pickled"
	f_atoms = open(filename_atoms, 'wb')
	filename_molecules = "molecules.pickled"
	f_molecules = open(filename_molecules, 'wb')
	pickle.dump(atoms,f_atoms) # write atoms to pickled file
	f_atoms.close()
	pickle.dump(molecules,f_molecules) # write molecules to pickled file
	f_molecules.close()
	
	if Debug_Pickling == 1:
		f_open_atoms = open(filename_atoms)
		f_open_molecules = open(filename_molecules)
		atoms_import_from_pickle = pickle.load(f_open_atoms)
		molecules_import_from_pickle = pickle.load(f_open_molecules)
		print "atom 1 ID:"
		print atoms_import_from_pickle[0].atomID
		print "atom 1 Positions:"
		print atoms_import_from_pickle[0].Pos
		print "molecule 1 ID:"
		print molecules_import_from_pickle[0].moleculeID
		print "molecule 1 atoms in backbone:"
		print molecules_import_from_pickle[0].AtomsBackbone

Overall_end_time = time.time()
f = open("Timing.LOG", "a")
f.write("Overall runtime: {}\n".format((Overall_end_time-Overall_start_time)))
f.close()