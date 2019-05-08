import numpy as np
import scipy as sp
import scipy.stats
import math

def ParseLAMMPSDataFile(filename):
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