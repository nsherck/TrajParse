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
from scipy.optimize import curve_fit
import stats_TrajParse as stats
import mdtraj as md
import mdtraj.reporters
import time
from pymbar import timeseries
import DataFileTemplates

global	numbermolecules
global	moleculeLength
global	beginatom
global	endatom
global	CalculateRg
global	MoleculeType
global  startStep
global  Number_Picked_Atoms # used to pick every ith bond along the backbone
global  Number_Skipped_Atoms
global  ATOM_TYPE_IGNORES_MOLECULES
global  ATOM_TYPE_IGNORES_BACKBONE
global  CGMappingBackbone
global  atomIndices
global  WrapCGCoordinates
global  Pickle_Objects
global  TimestepImportFrequency
global  Infiles
global  topfilename
global  WriteCGAtomRDFPairs
global  CGPairsIntermolecular
global  CGPairsIntramolecular
global  LammpsInputFilename
global  LammpsOutputScaleCoord 
global  CalculateCOM
global  CalculateRg
global  CalculateRee
global  CalculateBondBondCorrelationFunction
global  DoBootStrapping
global  PickSkip 
global  EveryNthAtom
global  RemoveFirstNatoms

#TESTING LOG

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
LammpsInputFilename = "system_EO19.data" # even if importing from .dcd need to have a LAMMPS topology file
startStep = 0 # Start on this index 
Read_Pickle = False
Pickle_Objects = False  # Do not pickle objects (also will not pickle if Read_Pickle is True, i.e. do not overwrite)
ShowFigures = False
''' DCD Specific '''
Infiles = ['chain_0.dcd']#,'chain_1.dcd']#,'chain_2.dcd','chain_3.dcd','chain_4.dcd']#,'nvt_production_output_wrapped_chain_1.dcd','nvt_production_output_wrapped_chain_2.dcd']
topfilename = 'EO19.pdb'
atomIndices = 135
Read_DCD    = True
TimestepImportFrequency = 1
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
CGMappingBackbone = [[3, 3, 1],[51, 3, 2],[3, 3, 1]] # for the first 72 atoms, take the three backbone atoms and map them to one!

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

global atoms
global molecules
atoms = []
molecules = []

# Create a class of atoms so that I can 
class Atom:
	def __init__(self, id_, mol_, type_, mass_, charge_):
		# the _ is there because id and type are built in to python and can't be overridden
		self.Pos = []
		self.MinImagePos = [] # minimum image with respect to a molecule, not global 
		self.ImageFlags = []
		self.Box = []
		self.neighList = []
		self.atomType = int(type_)
		self.atomID = int(id_)
		self.atomMol = int(mol_)
		self.atomMass = float(mass_)
		self.endFlag = 0
		self.TimeFrames = len(self.Pos)
		self.minImageFlag = 0 # if "1", then already minimum imaged
		self.atomCharge = float(charge_)
		self.angleList = []
		self.dihedralList = []


	def addCoord(self, pos, box, ImageFlag):
		'''Add a coordinate to this atom
		The expected box is of the form [(xlo,xhi),(ylo,yhi),(zlo,zhi)]'''
		for i in range(3):
			pos[i] = float(pos[i])
			box[i] = float(box[i])	
		self.Pos.append(pos)
		self.Box.append(box)
		self.ImageFlags.append(ImageFlag)

	def addNeighbor(self, neighbor):
		'''Specify an atom that is bonded to this one'''
		self.neighList.append(neighbor)
		
	def SwitchEndFlag(self, flag):
		''' turn the flag on "1" or off "0" '''
		if flag == 1:
			self.endFlag = 1
		elif flag == 0:
			self.endFlag = 0

# Create a class for molecules comprised of atoms or CG beads or both 
class Molecule:
	def __init__(self, mol_, type_):
		''' mol_ is the ID and type_ is "polymer_branched, polymer_star, etc '''
		# the _ is there because id and type are built in to python and can't be overridden
		self.Atoms = []
		self.CGAtoms = []
		self.AtomsBackbone = []
		self.CGAtomsBackbone = []
		self.CenterOfMass = []
		self.RadiusOfGyration = []
		self.EndtoEndVector = []
		self.moleculeBeginEndAtoms = []
		self.moleculeID = int(mol_)
		self.moleculeType = str(type_)
		self.molecularMass = 0.0
		self.moleculeBackboneBondVectors = [] # b_i = r_i+1 - r_i; a list of timesteps of molecular backbone bond vectors 
		self.moleculeBackboneBondVectorMagnitudes = []
		self.moleculeBackboneBondVectorMagnitudesTimeAverage = []
		self.moleculeBackboneBondVectorMagnitudesTimeStdDev = []
		self.moleculeBackboneBondVectorMagnitudesTimeVariance = []
		self.moleculeBackboneBondVectorMagnitudesTimeandChainLengthAverage = 0.0
		self.moleculeBackboneBondVectorMagnitudesTimeandChainLengthStdDev = 0.0
		self.moleculeBackboneBondVectorMagnitudesTimeandChainVariance = 0.0
		self.BondBondCorrelationFunction = []
		self.MonomerAtomsList = [] # contains the atomIDs that map to that monomer
		self.numberOfMonomers = 0 # Contains the number of monomers (=len(MonomerAtomsList))
		self.moleculeCGAtomIDs = [] # IDs of CGmonomers in this molecule
		
	#def CalculateMolecularMass(self)
	
	def addBeginEndAtoms(self, BeginEndAtoms):
		BeginAtom = BeginEndAtoms[0]
		EndAtom   = BeginEndAtoms[1]
		self.moleculeBeginEndAtoms.append(BeginAtom)
		self.moleculeBeginEndAtoms.append(EndAtom)	
			
	def addAtom(self, atom):
		'''Add an to the Molecule'''
		self.Atoms.append(atom)
	
	def parseAtoms(self, atoms):
		''' Build Polymer Backbone '''
		for atomID in self.Atoms:
			index = atomID - 1 
			if atoms[index].atomType not in ATOM_TYPE_IGNORES_BACKBONE:
				self.AtomsBackbone.append(atomID)
			else:
				pass
				
	def CalculateCenterOfMass(self, atoms):
		'''Calculate the Center of Mass of Molecule'''
		if len(self.Atoms) == 0:
			print "ERROR: No atoms in molecule list. Specify Atoms before calculating COM!"
			pass
		#else: 
			#print "Calculating Center of Mass"
		for i in range(len(atoms[0].Pos)): # iterate through time frames 
			tempmx = []
			tempmy = []
			tempmz = []
			TotalMass = 0
			for atomID in self.Atoms: # iterate through each atom in molecule
				index = atomID - 1 # python starts at 0
				tempmx.append(atoms[index].atomMass*atoms[index].Pos[i][0])
				tempmy.append(atoms[index].atomMass*atoms[index].Pos[i][1])
				tempmz.append(atoms[index].atomMass*atoms[index].Pos[i][2])
				if self.molecularMass == 0.0:
					TotalMass = TotalMass + atoms[index].atomMass
				else:
					TotalMass = self.molecularMass
			tempCOM = [(sum(tempmx)/TotalMass),(sum(tempmy)/TotalMass),(sum(tempmz)/TotalMass)]
			if self.molecularMass == 0.0: # set molecular mass
				self.molecularMass = TotalMass
			self.CenterOfMass.append(tempCOM)
			#print "Center of Mass: {}".format(tempCOM)
			
	def CGCalculateCenterOfMass(self, atoms, CGmap, cgatoms):
		'''Calculate the Center of Mass of the coarse-grained beads.'''
		''' NOTE: coarse-grained bead here is called a monomer '''
	
		#N.S. TODO: Currently the COM cg-ing only works by finding the neighbors of the atoms in the backbone. 
		#               Thus, if there are side-chains hanging off the backbone (i.e. methyl groups, etc.), it will
		#               not locate those. Need to generalize to instances where this might be the case. 
	
		# LOGIC:
		# (1) First, get backbone atoms in each CG bead (i.e. monomer here)
		# (2) Parse each atoms neighbor list to include atoms not in backbone, but being lumped into CG bead
		# (3) Finally, use vector operations to calculate COM for CG bead over all time frames. Should be quick, do not loop. 
		monomerCharge = 0.0
		MonomerMass = 1.0 # Set monomer mass to 1, replace below
		moleculeID = self.moleculeID
		moleculeCGAtomIDs = []
		
		if len(cgatoms) == 0:
			monomerID = 1
		else:
			monomerID = len(cgatoms) + 1
			
		if len(self.Atoms) == 0:
			print "ERROR: No atoms in molecule list, therefore no atoms to CG. Specify Atoms before calculating COM!"
			pass

		''' (1) Build atoms in Backbone going to 1 monomer. '''
		atomCnt = 0
		BackboneAtomIndexSlicing = []
		for i in CGmap: # generate slicing array
			MonomerType = i[2]
			
			for j in range((i[0]/i[1])):
				#print "j = {}".format(j)
				moleculeCGAtomIDs.append(monomerID)
				cgatoms.append(Atom(monomerID, moleculeID, MonomerType, MonomerMass, monomerCharge)) # make new monomer
				monomerID += 1
				cnt = j + 1 # range function starts from 0
				if atomCnt == 0:
					atomStart = 0
				atomEnd = atomStart + i[1] - 1 + 1 
				#print "AtomStart {}".format(atomStart)
				#print "AtomEnd   {}".format(atomEnd)
				BackboneAtomIndexSlicing.append(self.AtomsBackbone[atomStart:atomEnd]) # Slicing doesn't include the atomEnd element.
				atomStart = atomEnd
				atomCnt += i[1]
		self.moleculeCGAtomIDs = moleculeCGAtomIDs
		
		if Debug_CGing == 1:
			with open("BackboneAtomIndexSlicing.txt", "w") as f:
				for i,j in enumerate(BackboneAtomIndexSlicing):
					f.write("monomer {} has atoms {} \n".format(i,j))

		
		''' (2) Parse neighbor list of backbone atoms going to 1 monomer. '''
		# N.S. TODO: Need to generalize this 
		
		MonomerAtomsList = [] #List of List with atoms mapping to that monomer
		for i in BackboneAtomIndexSlicing: # pick out backbone atoms in single monomer.
			monomerTempIDList = [] # build up a modified list of atoms for each monomer that includes the backbone atoms and their neighbors.
			for atomID in i: # pick out atomID
				monomerTempIDList.append(atomID)
				index = atomID - 1
				atomNeighborIDs = atoms[index].neighList
				for neighborID in atomNeighborIDs:
					if int(neighborID) in self.AtomsBackbone:
						pass
					else:
						monomerTempIDList.append(int(neighborID))
			MonomerAtomsList.append(monomerTempIDList)
			self.MonomerAtomsList = MonomerAtomsList
			self.numberOfMonomers = len(self.MonomerAtomsList)
		
		if Debug_CGing == 1:
			with open("MonomerAtoms.txt", "w") as f:
				for i,j in enumerate(MonomerAtomsList):
					f.write("monomer {} has atoms {} \n".format(i,j))
				
		''' (3) Use vector calculus to compute COM. '''
		monomerCnt = 0
		for monomerAtoms in self.MonomerAtomsList:
			tempmx = []
			tempmy = []
			tempmz = []
			MonomerMass = []
			tempMonomerMass = 0.0
			PosM = 0.
			cnt = 0
			ListAtomIDs = []
			for atomIDs in monomerAtoms:
				ListAtomIDs.append(atomIDs)
				index = atomIDs-1 # python list starts from 0
				Pos = np.asarray(atoms[index].Pos)
				atomMass = atoms[index].atomMass 
				tempPosM = Pos*atomMass
				if cnt == 0:
					PosM = tempPosM*0.0
					PosM = np.add(PosM,tempPosM)
				else:
					PosM = np.add(PosM,tempPosM)
				tempMonomerMass += atomMass
				cnt +=1
			MonomerMass.append(tempMonomerMass)
			#print "MonomerMasses {}".format(MonomerMass)	
			CGCOM = np.divide(PosM,tempMonomerMass).tolist()
			
			tempBox = atoms[index].Box
			imageFlags = ['N/A','N/A','N/A']
			#N.S. TODO: Need to add in the ability to have image flags for the CG particles!
			cgatoms[(self.moleculeCGAtomIDs[monomerCnt]-1)].Pos = CGCOM # update the atoms CGCOM
			cgatoms[(self.moleculeCGAtomIDs[monomerCnt]-1)].Box = tempBox
			cgatoms[(self.moleculeCGAtomIDs[monomerCnt]-1)].ImageFlags = imageFlags
			monomerCnt += 1
			
			if Debug_CGing == 1:
				with open("MonomerCGCOM.txt", "w") as f:
					f.write("Atom IDs {} \n".format(ListAtomIDs))
					for i,j in enumerate(CGCOM):
						f.write("timestep {} & COM {} \n".format(i,j))
		
		#print "cgatom1 COM"
		#print cgatoms[0].Pos
		
	def CalculateRadiusOfGyration(self, atoms):
		''' Calculate the Radius of Gyration '''
		# Need to calculate Center of Masses first!
		if len(self.CenterOfMass) == 0:
			print "ERROR: No Center of Masses. Calculate COM before calculating Rg!"
			pass 
			#print "Calculating Radius of Gyration"
			
		for i in range(len(atoms[0].Pos)): # iterate through time frames 
			tempdTot = [] # reset tempdTot
			for atomID in self.Atoms: # iterate through each atom in molecule
				index = atomID - 1 # python list starts at 0
				temp_mass = atoms[index].atomMass
				tempdx = (atoms[index].Pos[i][0]-self.CenterOfMass[i][0])**2
				tempdy = (atoms[index].Pos[i][1]-self.CenterOfMass[i][1])**2
				tempdz = (atoms[index].Pos[i][2]-self.CenterOfMass[i][2])**2
				tempdTot.append(temp_mass*(tempdx + tempdy + tempdz))
			if self.molecularMass == 0.0:
				print "ERROR: Molecular Mass = 0.0"
			else:
				TotalMass = self.molecularMass
			# Calculate the Rg for this Time Frame
			tempRg = math.sqrt(float(sum(tempdTot)/TotalMass))
			self.RadiusOfGyration.append(tempRg)
			#print tempRg
			
	def CalculateEndtoEndVector(self, atoms):
		''' Calculates the end-to-end vector for each molecule '''
		BeginAtom_ = self.moleculeBeginEndAtoms[0]
		EndAtom_   = self.moleculeBeginEndAtoms[1]
		temp_dist = []
		#print atoms[BeginAtom_].Pos[0]
		#print atoms[EndAtom_].Pos[0]
		for i in range(len(atoms[(BeginAtom_-1)].Pos)): # iterate through each time frame 
			temp_dist = [(i - j)**2 for i, j in zip(atoms[(BeginAtom_-1)].Pos[i], atoms[(EndAtom_-1)].Pos[i])]
			#print "temp_dist"
			#print temp_dist
		
			temp_dist = math.sqrt(sum(temp_dist))
			self.EndtoEndVector.append(temp_dist)
			#print "temp_dist"
			#print temp_dist
				
	def MolcularMinimumImage(self, atoms):
		''' Calculate the minimum image of the molecule '''
		# Uses the beginning atom on the molecule as the atom to start from.
		# Note: the Pos list contains all positions for atom over timesteps. 
		# USING THIS FEATURE IS INCORRECT FOR CALCULATING Rg OR Ree. NEED TO 
		# 	USE UNWRAPPED COORDINATES.
		Lbox = []
		Posref = []
		cnt = 0
		for atomID in self.AtomsBackbone:
			Index = atomID - 1
			Pos0 = np.asarray(atoms[Index].Pos)
			atoms[Index].MinImagePos = atoms[Index].Pos
			
			if cnt == 0: # if first atom, set as first position
				atoms[Index].minImageFlag = 1
				Lbox = atoms[Index].Box # set the box side lengths (x,y,z)
			else:
				pass
			cnt += 1
			
			for neigh_atomID in atoms[Index].neighList:
				#print neigh_atomID
				neigh_index = int(neigh_atomID) - 1
				if atoms[int(neigh_index)].minImageFlag != 1: # not min. imaged yet					 
					Pos1 = np.asarray(atoms[int(neigh_index)].Pos)
					Posref = np.subtract(Pos1, Pos0)
					temp = np.absolute(np.round(np.divide(Posref,Lbox)))
					temp = np.multiply(Lbox,temp)
					Posref = np.subtract(Posref,temp)
					atoms[int(neigh_index)].MinImagePos = (np.add(Pos0,Posref)).tolist()
					atoms[int(neigh_index)].minImageFlag = 1
					Pos0 = atoms[int(neigh_index)].MinImagePos # Update reference position
				else: # already min. imaged
					pass
		
	def CalculateMolecularBondVectors(self,atoms):
		''' Calculates a list of list. Each list contains the molecular bond vectors at that
				timestep. Where bond vector is defined b_i = r_i+1 - r_i '''
		bond_temp = []
		bond_magnitude_temp = [] # magnitude of the bond-bond vector
		# The variable "temp_reducedAtomsBackbone" allows one to pick every so many bonds
		#	instead of using every bond in the polymer chain.
		temp_reducedAtomsBackbone = []
		cnt = 0
		FlagSkip = False
		if Number_Picked_Atoms == -9999:
			temp_reducedAtomsBackbone = self.AtomsBackbone
		elif PickSkip == True:
			for i in self.AtomsBackbone:
				if cnt == Number_Picked_Atoms:
					cnt = 0
					FlagSkip = True
				if FlagSkip == True and cnt == Number_Skipped_Atoms:
					cnt = 0
					FlagSkip = False
					temp_reducedAtomsBackbone.append(i)
				elif FlagSkip == True:
					cnt += 1
					continue
				else:
					temp_reducedAtomsBackbone.append(i)
				cnt += 1
		else: 
			temp = self.AtomsBackbone
			for i in RemoveFirstNatoms:
				del temp[i]
			temp_reducedAtomsBackbone = self.AtomsBackbone[0::EveryNthAtom]
		#print "reduced atoms backbone"
		#print temp_reducedAtomsBackbone
			
			
		max = len(temp_reducedAtomsBackbone) - 1
		for cnt,atomID in enumerate(temp_reducedAtomsBackbone):
			if cnt == max: # check if last atom
				continue
			else:
				atom_1_index = atomID - 1
				atom_2_index = (temp_reducedAtomsBackbone[(cnt+1)] - 1)
				atom_1_Pos = atoms[atom_1_index].Pos
				atom_2_Pos = atoms[atom_2_index].Pos
				#print "atomID"
				#print atomID
				#print (self.AtomsBackbone[(cnt+1)] - 1)
				#print atoms[atom_2_index].atomID
				#print atoms[atom_1_index].neighList
				if str(atoms[atom_2_index].atomID) in atoms[atom_1_index].neighList:
					atom_2_Pos = np.asarray(atom_2_Pos)
					atom_1_Pos = np.asarray(atom_1_Pos)
					temp = np.subtract(atom_2_Pos, atom_1_Pos)
					temp_magnitude = np.sqrt(np.sum(np.multiply(temp,temp),axis=1)) # Could have taken absValue(temp)
					bond_temp.append(temp.tolist())
					bond_magnitude_temp.append(temp_magnitude.tolist())
				else: # originally, this was designed for only bonded atoms, now made more general.
					atom_2_Pos = np.asarray(atom_2_Pos)
					atom_1_Pos = np.asarray(atom_1_Pos)
					temp = np.subtract(atom_2_Pos, atom_1_Pos)
					temp_magnitude = np.sqrt(np.sum(np.multiply(temp,temp),axis=1)) # Could have taken absValue(temp)
					bond_temp.append(temp.tolist())
					bond_magnitude_temp.append(temp_magnitude.tolist())				
					#print "ERROR: ATOM ID {} NOT IN NEIGHBOR LIST OF ATOM {}.".format(atoms[atom_2_index].atomID, atomID)
					#print "Code will continue, but Bond vectors are likely incorrect!"
					#print "Ignore if you are skipping bonds!"
					#continue
		self.moleculeBackboneBondVectors = bond_temp
		self.moleculeBackboneBondVectorMagnitudes = bond_magnitude_temp
		self.moleculeBackboneBondVectorMagnitudesTimeAverage = np.average(np.asarray(bond_magnitude_temp),axis=1).tolist()
		#print "molecule backbone average"
		#print self.moleculeBackboneBondVectorMagnitudesTimeAverage
		#print "Length"
		#print len(self.moleculeBackboneBondVectorMagnitudesTimeAverage)
		self.moleculeBackboneBondVectorMagnitudesTimeStdDev = np.std(np.asarray(bond_magnitude_temp),axis=1).tolist()
		self.moleculeBackboneBondVectorMagnitudesTimeVariance = np.var(np.asarray(bond_magnitude_temp),axis=1).tolist()
		self.moleculeBackboneBondVectorMagnitudesTimeandChainLengthAverage = np.asarray(bond_magnitude_temp).mean()
		self.moleculeBackboneBondVectorMagnitudesTimeandChainLengthStdDev = np.asarray(bond_magnitude_temp).std()
		self.moleculeBackboneBondVectorMagnitudesTimeandChainVariance = np.asarray(bond_magnitude_temp).var()
		
	def CalculateBondBondCorrelationFunction(self, atoms):
		''' Calculates the bond-bond correlation function along the polymer backbone '''
		# This uses unwrapped coordinates
		BondBondCorrelation_Ignore_AtomTypes = [] # can specify atom types to ignore
		BondBondCorrelation = []
		LengthBondVectorList = len(self.moleculeBackboneBondVectors)
		#print "Length bond vector list is {}".format(LengthBondVectorList)
		for index_1,bondVec_1 in enumerate(self.moleculeBackboneBondVectors):
			for index_2,bondVec_2 in enumerate(self.moleculeBackboneBondVectors):
				if index_2 < index_1: # Do not double count!
					continue
				#Uncomment the next two lines for quick trouble-shooting
				#if index_2 > 2 or index_1 > 0:
				#	continue
				bondVec_1 = np.array(bondVec_1)
				bondVec_2 = np.asarray(bondVec_2)
				#print "bond Vector 1"
				#print bondVec_1
				#print len(bondVec_1)
				MagVec_1 = np.asarray(self.moleculeBackboneBondVectorMagnitudes[index_1])
				#print "magnitude bond vector 1"
				#print MagVec_1
				#print len(MagVec_1)
				MagVec_2 = np.asarray(self.moleculeBackboneBondVectorMagnitudes[index_2])
				temp_BondDistance = abs(index_2 - index_1)
				temp_BBCorrelation = np.multiply(bondVec_1,bondVec_2)
				temp_BBCorrelation = np.sum(temp_BBCorrelation,axis=1)
				temp_BBMagnitude = np.multiply(MagVec_1,MagVec_2)
				temp_BBCorrelation = np.divide(temp_BBCorrelation,temp_BBMagnitude)
				# N.S. TODO: Extend to work along the chain backbone, i.e. make chain length dependent
				temp_avg = np.average(temp_BBCorrelation)
				if index_1 == 0:
					BondBondCorrelation.append(temp_avg.tolist())
				else:
					BondBondCorrelation[temp_BondDistance] = (BondBondCorrelation[temp_BondDistance]+temp_avg)/2. 
		self.BondBondCorrelationFunction = BondBondCorrelation
		#print "Length of Bond-Bond Correlation Function is {}".format(len(BondBondCorrelation))
	
	def PlotBondBondCorrelationFunction(self):	
		''' Plots the bond-bond correlation function '''
		print "Not in function"

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
	
def ParseInputFile(filename):
	''' Parses a LAMMPS data file to build topology '''
	#First must generate a list of atoms with their respective IDs and atom types
	print 'Reading in the input file: {}'.format(filename)
	infile = str(filename)
	with open(infile, 'rb') as f:
		dumplist = f.read().split('Atoms')
		dumplist_mass = dumplist[0].split('Masses')
		dumplist_atom = dumplist[1].split('Bonds')
		dumplist_temp = dumplist[1].split('Bonds')
		dumplist_bonds = dumplist_temp[1].split('Angles')
	masslist = dumplist_mass[1]
	atomlist = dumplist_atom[0]
	bondlist = dumplist_bonds[0]
	del dumplist
	
	
	if Debug_ParseData == 1:
		print '************ Atom List ***********'
		print atomlist
		print '************ Bond List ***********'
		print bondlist
		print '************ Mass List ***********'
		print masslist
		
	
	
	for i in atomlist.split("\n"):
		if Debug_ParseData == 1: 
			print "All the lines in Atom List"
			print i
		# Split the lines in atomlist by spaces
		line = i.split()
		
		# Begin constructing atoms
		if len(line) == 0:
			pass
		else:
			id_ = line[0]
			mol_ = line[1]
			type_ = line[2]
			charge_ = line[3]
			# Find the mass value
			if int(type_) in ATOM_TYPE_IGNORES_MOLECULES:
				continue
			else:
				for i in masslist.split("\n"):
					line_mass = i.split()
					#print "line_mass {}".format(line_mass)
					if len(line_mass) == 0:
						pass
					elif line_mass[0] == type_ and line_mass[0] not in ATOM_TYPE_IGNORES_MOLECULES:
						mass_ = line_mass[1]
					else:
						pass
				# define instance of an atom
				atoms.append(Atom(id_, mol_, type_, mass_, charge_))
	atoms.sort(key=lambda atom: atom.atomID) # sort atoms by ID	
	

	
	''' Put Atoms into Molecules '''
	IgnoredMolecules = 0
	#initialize molecule of type 1
	molecules.append(Molecule(1, "polymer_linear"))
	for atom in atoms:
		tempType = atom.atomType
		tempMol  = atom.atomMol
		if tempType not in ATOM_TYPE_IGNORES_MOLECULES:
			AtomPlaced = 0
			for molecule in molecules:
				if molecule.moleculeID == tempMol: # add to existing molecule
					molecule.addAtom(atom.atomID)
					AtomPlaced = 1
			if AtomPlaced == 0: # Create new molecule
					molecules.append(Molecule(tempMol, MoleculeType))
					molecules[-1].addAtom(atom.atomID) # add atom to molecule list
			AtomPlaced = 0
		else:
			IgnoredMolecules = IgnoredMolecules + 1
	
	print "Number of molecules found in system: {}".format(len(molecules)) 
	print "Number of excluded molecules in system: {}".format(IgnoredMolecules)
	
	#Molecule_1 = Molecule(1,1)
	#for i in atoms: 
	#	tempID = i.atomID
	#	tempChemistry = i.atomChemistry
		# Checks to see if a Hydrogen, otherwise considered a "backbone" atom.
		# This is okay for PEO, but maybe not more complicated polymer chemistries. 
		# N.S. TODO: Generalize this functionality!
	#	if tempChemistry != "H":
	#		Molecule_1.addAtom(tempID)
	#	else:
	#		pass
	if Debug_ParseData == 1:
		print "Atoms in the first molecule:"
		print molecules[0].Atoms
		
	
	''' Add in atom bonding '''
	print "Calculating bonding!"
	
	for i in atoms:
		tempID = i.atomID
		if i.atomType not in ATOM_TYPE_IGNORES_MOLECULES:
			for j in bondlist.split("\n"):
				line = j.split()
				if len(line) < 3:
					pass
				else:
					if int(tempID) == int(line[2]):
						i.addNeighbor(line[3])
					if int(tempID) == int(line[3]):
						i.addNeighbor(line[2])
	
	if Debug_ParseData == 1:
		print "atom 1"
		print atoms[0].atomID
		print atoms[0].atomMol
		print atoms[0].atomType
		print atoms[0].atomMass
		print atoms[0].atomCharge
		print atoms[0].neighList

		print "last atom"
		print atoms[1].atomID
		print atoms[1].atomMol
		print atoms[1].atomType
		print atoms[1].atomMass
		print atoms[1].atomCharge
		print atoms[1].neighList
	
	if Debug_ParseData == 1:
		print "Atoms in Neighbor list for Atom 1"
		print atoms[0].neighList
		print "Atoms in Neighbor list for Atom 2"
		print atoms[1].neighList
	
	print "Building Backbone atoms."
	for i in molecules:
		i.parseAtoms(atoms)
		temp_StartAtoms = i.AtomsBackbone[0]
		temp_EndAtoms = i.AtomsBackbone[-1]
		i.moleculeBeginEndAtoms = [temp_StartAtoms, temp_EndAtoms]
		
	if Debug_ParseData == 1:
		print "molecule 1 backbone atoms:"
		print molecules[0].AtomsBackbone
		print "molecule 2 backbone atoms:"
		print molecules[1].AtomsBackbone
	
	
	if Debug_ParseData == 1:
		for i in molecules:
			print "Beginning and ending atom IDs in molecule {}:".format(i.moleculeID)
			print i.moleculeBeginEndAtoms
		
	
	return atoms, molecules
	
def analyze_dump(infile, FileFormat, style='beadspring', title='', POLY_END_TYPE = 1, POLY_MID_TYPES = [2], COLLOID_TYPE = 4, TYPE_IGNORES = [20,84,85], id1 = None, id2 = None, nbins=100, expectedPotentialX=None, expectedPotentialY=None, minE=None, infE = None, show=True, save=True, noVolume=False): #It would make a lot more sense to just input which bond distances you want by atom type...
	'''Analyze an atom-type dump file
	INPUT:
		infile: The name of an atom-type dump file generated by LAMMPS
		style: 'polymer' or 'colloid', style of the contents of the dump file
		POLY_END_TYPE and POLY_MID_TYPES: only used if style is 'polymer',
			these are the respective types of the atoms that make up the polymer.
			POLY_MID_TYPES supports multiple types in list form.'''
	
	# TODO: Get the Atom masses from the data file
	#atom_Masses = [[5, 12.01],[20, 1.008],[63, 16.00],[84, 15.9994],[85, 1.008]]
	
	if Read_LAMMPSTraj == True:
		''' Find out molecular topology '''
		# fills in atom bonding
		if ReadDataFile == 1:
			atoms, molecules = ParseInputFile(LammpsInputFilename)
		else: 
			pass
		import pickle
		if 'pickle' in infile: # this is already a pickled trajectory file
			with open(infile, 'rb') as f:
				dists = pickle.load(f)
			print 'Number of data points: ', len(dists)
		else: # else parse the data
			if type(POLY_MID_TYPES) == int:
				POLY_MID_TYPES = [POLY_MID_TYPES]
			if POLY_END_TYPE in POLY_MID_TYPES:
				print 'ERROR: You specified that the end type of the polymer was in POLY_MID_TYPES.'
				raise ValueError
			if type(TYPE_IGNORES) == int:
				TYPE_IGNORES = [TYPE_IGNORES]
			
			#First must generate a list of atoms with their respective IDs and atom types
			print 'Creating atom list'
			with open(infile, 'rb') as f:
				dumplist = f.read().split('ITEM: TIMESTEP')
			del dumplist[0] # this is an empty string
			
			''' OLD STUFF '''
			'''
			#print dumplist
			
			for i in dumplist[0].split('ITEM: ATOMS id mol type xsu ysu zsu ix iy iz')[1].split('\n'): 
			# This is a terrible way of looping through the lines that have the initial position info I need
				#print repr(i)
				line = i.split()
				#print line
				if len(line) < 1:
					continue
				id_ = line[0]
				mol_ = line[1]
				type_ = line[2]
				# Add in Atom masses
				mass_ = 0
				#print "type_"
				#print type_
				for i in atom_Masses:
					if i[0]==int(line[2]) :
						mass_ = i[1]
				if mass_ == 0:
					print "WARNING: Atom Mass not found. Defaulting to 1.0"
					mass_ = 1.0
				atoms.append(Atom(id_, mol_, type_, mass_))
			atoms.sort(key=lambda atom: atom.atomID) #Sort atoms by ID
			'''
			
			''' Nick Check '''
			#print "Atom Masses"
			#for atom in atoms:
			#	print atom.atomMass
				
			#Fill atoms with position data
			print 'Filling position values'
			skipCount = 0
			for indx,timestepData in enumerate(dumplist):
				timestep = dumplist[indx].split('\n')[1]
				if int(indx)<int(startStep):
					skipCount+=1
					continue
				temp = timestepData.split('ITEM: ATOMS id mol type xsu ysu zsu ix iy iz')[1].split('\n')
				temp2 = timestepData.split('ITEM: ATOMS id mol type xsu ysu zsu ix iy iz')[0].split('ITEM: BOX BOUNDS pp pp pp')[1].split('\n')
				box = []
				for i in temp2:
					if i != '':
						box.append(i.split())
						for i in range(len(box)):
							for j in range(len(box[i])):
								box[i][j] = float(box[i][j])

				if [] in box:
					del box[0]
				''' Nick Check '''
				#print indx
				#print temp
				#print box
				for atom_data in temp:
					# print repr(atom_data)
					atom_data = atom_data.split()
					if len(atom_data) == 0:
						continue
					id_ = int(atom_data[0]) # id
					mol_ = int(atom_data[1]) # molecule
					type_ = int(atom_data[2]) # atom type
					if type_ in ATOM_TYPE_IGNORES_MOLECULES: # SKIPS MOLECULES
						continue
					Pos = [float(atom_data[3]), float(atom_data[4]), float(atom_data[5])]
					ImageFlags = [int(atom_data[6]), int(atom_data[7]), int(atom_data[8])]
					#ImageFlags = [0,0,0]
					''' Nick Check '''
					#print Pos
					for atom in atoms:
						if atom.atomID == id_:
							for i in range(3):
								Pos[i] = (box[i][1] - box[i][0])*Pos[i]
							box_temp = []
							for i in box:
								box_temp.append(i[1]-i[0])
							atom.addCoord(Pos, box_temp, ImageFlags)
							break
					else:
						print ("ID not found {}".format(id_))
						print ("{}".format(timestepData.split('ITEM: ATOMS id mol type xsu ysu zsu ix iy iz')[0]))

			print ('Number of data points per atom (timesteps recorded): {}'.format(len(atoms[0].Pos)))
			print ('Number of points skipped: {}'.format(skipCount))
			
	elif Read_DCD == True:
		if ReadDataFile == 1: # Read in a LAMMPS Data File 
			atoms, molecules = ParseInputFile(LammpsInputFilename)
		else: 
			pass
		
		aIndices = range(atomIndices)
		''' Load in a DCD Traj. file. '''
		# Requires a .pdb file (for topology)
		# Added 2019.02.21 by N.S.
		print ("Loading in a .DCD trajectory file.")
		traj = md.load(Infiles,top=topfilename,atom_indices=aIndices)
		#print "trajectory frame 0"
		#print traj.xyz[0]
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
			#print "length atoms is {}".format(len(atoms))
			cnt +=1
			for index,atom in enumerate(atoms):
				#print index
				atom.Pos.append(frame[index])
		#print cnt
		
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
			
			
	if CalculateCOM == True:
		print ("Calculating Molecular Center of Masses.")
		for index,Molecule in enumerate(molecules):
			#print index
			start_time_MinImage = time.time()
			Molecule.MolcularMinimumImage(atoms)
			end_time_MinImage = time.time()
			#print 'COM'
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
			molecule.CalculateMolecularBondVectors(atoms)
		
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
			molecule.CGCalculateCenterOfMass(atoms, CGMappingBackbone, cgatoms)
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
			
if __name__ == "__main__":
	analyze_dump('dump.coords_u_s.dat', FileFormat='PDB', style='beadspring', title='', POLY_END_TYPE = 1, POLY_MID_TYPES = [2], COLLOID_TYPE = 4, TYPE_IGNORES = [84,85,20], id1 = None, id2 = None, nbins=100, expectedPotentialX=None, expectedPotentialY=None, minE=None, infE = None, show=True, save=True, noVolume=False)