import numpy as np
import scipy as sp
import scipy.stats
import math
from TrajParse_Classes import Atom, Molecule 
import mdtraj as md


def MoleculeType(molecule):
	''' Logic behind molecule types '''
	atomsInMolecule = len(molecule.Atoms)
	if atomsInMolecule == 1:
		temp = 'atomic'
	elif atomsInMolecule <= 10:
		temp = 'small_molecule'
	elif atomsInMolecule > 10:
		temp = 'polymer_linear'
	molecule.moleculeType = temp
	

def ParseLAMMPSDataFile(filename, Debug_ParseData, ATOM_TYPE_IGNORES_MOLECULES, ATOM_TYPE_IGNORES_BACKBONE):
	''' Parses a LAMMPS data file to build topology '''
	atoms = []
	molecules = []
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
	molecules.append(Molecule(1))
	molecules[0].moleculeType = 'polymer_linear'
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
					molecules[-1].moleculeType = 'polymer_linear'
					molecules[-1].addAtom(atom.atomID) # add atom to molecule list
			AtomPlaced = 0
		else:
			IgnoredMolecules = IgnoredMolecules + 1
	
	print "Number of molecules found in system: {}".format(len(molecules)) 
	print "Number of excluded molecules in system: {}".format(IgnoredMolecules)
	
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
	ATOM_NAMES_IGNORES_BACKBONE = []
	for i in molecules:
		i.parseAtoms(atoms, ATOM_TYPE_IGNORES_BACKBONE, ATOM_NAMES_IGNORES_BACKBONE)
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
	
def ParsePDBDataFile(filename, mol2filename, Debug_ParseData, IGNORE_WATER, ATOM_TYPE_IGNORES, ATOM_TYPE_IGNORES_BACKBONE, ATOM_NAMES_IGNORES_BACKBONE):
	''' Parses a .PDB and .mol2 data files to build topology '''
	# Assumes the solvent comes last
	# Assumes the bonding information comes from .mol2 file (with same atom indexing!)
	#	- in the case of PAA it goes: PAA, Na+, water
	# No charges currently available
	# Assumes bonding lays between @<TRIPOS>SUBSTRUCTUTRE, @<TRIPOS>SUBSTRUCTURE
	#	- should be straightforward to generalize
	# Assumes that the first/last atom in the backbone list are the first/last atom at the begin/end of chain (for Ree)
	
	#Debug_ParseData = True

	ElementDictionary ={
				"carbon": "c",
				"hydrogen": "h",
				"oxygen": "o",
				"nitrogen": "n",
				"sodium": "na+",
				"virtual site": "vs",
				}
	
	AtomicMassDictionary ={
					"c":  12.01,
					"h": 1.008,
					"o": 16.00,
					"n": 14.01,
					"na+": 22.989,
					"vs": 0.0,
					}
	
	atoms = []
	molecules = []
	#First must generate a list of atoms with their respective IDs and atom types
	print 'Reading in the input file: {}'.format(filename)
	infile = str(filename)
	topology = md.load(filename).topology # mdtraj loads in the topology
	
	''' Generate atoms '''
	molecule_cnt = 0
	atom_cnt = 0
	top_atomindexlist = [] # keeps the atom indexes from the topology file
	chain_id_prior = 0
	for i,chain in enumerate(topology.chains):
		molecule_cnt += 1
		chain_id_current = chain.index
		for j,residue in enumerate(chain.residues):
			Flag_include = 'Yes'
			if chain_id_current != chain_id_prior and residue.is_water == True:
				molecule_cnt -= 1 # switch to water, over-counts by 1 the first time
			
			if residue.is_water == True and IGNORE_WATER == True: # parse out water
				Flag_include = 'NO'
			elif residue.is_water == True and IGNORE_WATER == False:
				Flag_inlcude = 'YES'
				molecule_cnt += 1
				
			for k,atom in enumerate(residue.atoms):		
				if  ElementDictionary[str(atom.element)] in ATOM_TYPE_IGNORES: # parse out atom type ignores
					Flag_include = 'NO'
				
				if Flag_include == 'NO':
					pass
				else:
					''' initialize atom object '''
					atom_cnt += 1
					id_ = int(atom_cnt)
					mol_ = molecule_cnt
					type_ = (str(ElementDictionary[str(atom.element)])).lower()
					charge_ = 0.0 #no charging information at this time
					mass_ = AtomicMassDictionary[str(type_)]
					atoms.append(Atom(id_, mol_, type_, mass_, charge_)) # append atom object to list
					atoms[-1].atomName = atom.name
					top_atomindexlist.append([atom_cnt,atom.index]) # info. to find atom in topology object
		chain_id_prior = chain_id_current
	atoms.sort(key=lambda atom: atom.atomID) # sort atoms by ID, should be in order	
	if Debug_ParseData == True:
		print ("Attributes of Atom 1:")
		print atoms[0].atomMol
		print atoms[0].atomType
		print ("top_atomindexlist")
		print top_atomindexlist
		
	''' Put Atoms into Molecules '''
	# TODO: Put in functionality to count ignored molecules
	IgnoredMolecules = 0
	#initialize molecule of type 0
	molecules.append(Molecule(1))
	for atom in atoms:
		tempType = atom.atomType
		tempMol  = atom.atomMol
		AtomPlaced = 0
		for molecule in molecules:
			if molecule.moleculeID == tempMol: # add to existing molecule
				molecule.addAtom(atom.atomID)
				AtomPlaced = 1
		if AtomPlaced == 0: # Create new molecule
				molecules.append(Molecule(tempMol))
				molecules[-1].addAtom(atom.atomID) # add atom to molecule list
		AtomPlaced = 0
	
	for index, molecule in enumerate(molecules): # Type the molecule
		MoleculeType(molecule)
		if len(molecule.Atoms) == 0: # check if no atoms
			del molecules[index] # delete molecule if no atoms
			
	print "Number of molecules found in system: {}".format(len(molecules)) 
	print "Number of excluded molecules in system: {}".format(IgnoredMolecules)
	
	if Debug_ParseData == 1:
		print "Atoms in the first molecule:"
		print molecules[0].Atoms
	
	''' Add in atom bonding '''
	print "Calculating bonding!"
	
	with open(mol2filename, 'rb') as f: # read in .mol2 file
		dumplist = f.read().split('@<TRIPOS>BOND')
		dumplist_bonding = dumplist[1].split('@<TRIPOS>SUBSTRUCTURE')
	bondlist = dumplist_bonding[0]
	
	for i,atom in enumerate(atoms):
		tempID = atom.atomID
		for j in bondlist.split("\n"):
			line = j.split()
			if len(line) < 3:
				pass
			else:
				tempID = top_atomindexlist[i][1] + 1
				#print "tempID"
				#print tempID
				#print line
				#print line[2]
				#print line[3]
				if int(tempID) == int(line[1]):
					atom.addNeighbor(line[2])
				if int(tempID) == int(line[2]):
					atom.addNeighbor(line[1])
	
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
	
	''' Get the polymer backbone atoms '''
	# Only for moleculeType == polymer_linear!
	print "Building Backbone atoms."
	for molecule in molecules:
		if 'polymer_linear' in str(molecule.moleculeType): 
			molecule.parseAtoms(atoms, ATOM_TYPE_IGNORES_BACKBONE, ATOM_NAMES_IGNORES_BACKBONE)
			temp_StartAtoms = molecule.AtomsBackbone[0]
			temp_EndAtoms = molecule.AtomsBackbone[-1]
			molecule.moleculeBeginEndAtoms = [temp_StartAtoms, temp_EndAtoms]
		else:
			pass
		
	if Debug_ParseData == 1:
		print "molecule 1 backbone atoms:"
		print molecules[0].AtomsBackbone
		#print "molecule 2 backbone atoms:"
		#print molecules[1].AtomsBackbone
	
	
	if Debug_ParseData == 1:
		for i in molecules:
			print "Beginning and ending atom IDs in molecule {}:".format(i.moleculeID)
			print i.moleculeBeginEndAtoms
		
	
	return atoms, molecules, top_atomindexlist
	