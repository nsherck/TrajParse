import numpy as np
import scipy as sp
import scipy.stats
import math

# Create a class of atoms so that I can 
class Atom:
	def __init__(self, id_, mol_, type_, mass_, charge_):
		# the _ is there because id and type are built in to python and can't be overridden
		self.Pos = []
		self.MinImagePos = [] # minimum image with respect to a molecule, not global 
		self.ImageFlags = []
		self.Box = []
		self.neighList = []
		self.atomType = type_
		self.atomID = int(id_)
		self.atomMol = int(mol_)
		self.atomMass = float(mass_)
		self.atomName = 'N/A' # name from PDB file, not always used
		self.endFlag = 0
		self.TimeFrames = len(self.Pos)
		self.minImageFlag = 0 # if "1", then already minimum imaged
		self.atomCharge = float(charge_)
		self.angleList = []
		self.dihedralList = []
		self.chemistry = 'N/A'
		self.solvent = False
		self.residue = 0


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
	def __init__(self, mol_):
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
		self.moleculeType = 'N/A'
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
	
	def parseAtoms(self, atoms, ATOM_TYPE_IGNORES_BACKBONE, ATOM_NAMES_IGNORES_BACKBONE):
		''' Build Polymer Backbone '''
		for atomID in self.Atoms:
			index = atomID - 1 
			if atoms[index].atomType not in ATOM_TYPE_IGNORES_BACKBONE:
				Flag_pass = False
				for i_excl in ATOM_NAMES_IGNORES_BACKBONE:
					if i_excl in atoms[index].atomName:
						Flag_pass = True
			if Flag_pass == False:	
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
			
	def CGCalculateCenterOfMass(self, atoms, CGmap, cgatoms, Debug_CGing):
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
		
	def CalculateMolecularBondVectors(self,atoms, Number_Picked_Atoms, Number_Skipped_Atoms, RemoveFirstNatoms, EveryNthAtom, PickSkip):
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