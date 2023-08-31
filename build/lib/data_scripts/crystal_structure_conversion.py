# %%
# import ase
from pymatgen.core import Structure, Element
from ase import Atoms as AseAtoms
from pymatgen.io.ase import AseAtomsAdaptor as pase
from jarvis.core.atoms import Atoms as JarvisAtoms
# %%
def pymatgen_to_ase(pmg_structure):
    """Convert a PymatgenStructure object to an AseAtoms object.

  Args:
      pmg_structure (pymatgen.Structure): The pymatgen structure object to convert.

  Returns:
      ase.Atoms: The ASE atom object.
    """
    ase_atoms = pase.get_atoms(pmg_structure)
    return ase_atoms

def ase_to_pymatgen(ase_atoms):
    """Convert an AseAtoms object to a PymatgenStructure object.

    """
    pmg_structure = pase.get_structure(ase_atoms)
    return pmg_structure

def ase_to_jarvis(ase_atoms="", cartesian=True):
    """Convert AseAtoms to JarvisAtoms."""
    return JarvisAtoms(
        lattice_mat=ase_atoms.get_cell(),
        elements=ase_atoms.get_chemical_symbols(),
        coords=ase_atoms.get_positions(),
        cartesian=cartesian,
    )
    
def jarvisP_to_ase(jarvis_atoms):
    """Convert a Jarvis-core Atoms object to an ASE atom object.

    Args:
        jarvis_atoms (jarvis.core.atoms.Atoms): The Jarvis-core Atoms object to convert.

    Returns:
        ase.Atoms: The ASE atom object.
    """
    species = [s.strip() for s in jarvis_atoms.elements]
    coords = jarvis_atoms.frac_coords
    lattice = jarvis_atoms.lattice_mat
    ase_atoms = AseAtoms(symbols=species, 
                         scaled_positions=coords, 
                         cell=lattice,
                         pbc = True)
    return ase_atoms

def pymatgen_to_jarvis(pmg_structure):
    """Convert a pymatgen structure object to a Jarvis-core Atoms object.

    Args:
        pmg_structure (pymatgen.Structure): The pymatgen structure object to convert.

    Returns:
        jarvis.core.atoms.Atoms: The Jarvis-core Atoms object.
    """
    species = pmg_structure.species
    coords = pmg_structure.cartesian_coords
    lattice = pmg_structure.lattice.matrix
    jarvis_atoms = JarvisAtoms(species=species, coords=coords, lattice=lattice)
    return jarvis_atoms
    

def jarvis_to_pymatgen(jarvis_atoms):
    """Convert a Jarvis-core Atoms object to a pymatgen structure object.

    Args:
        jarvis_atoms (jarvis.core.atoms.Atoms): The Jarvis-core Atoms object to convert.

    Returns:
        pymatgen.Structure: The pymatgen structure object.
    """
    # species = [Element.from_symbol(s) for s in jarvis_atoms.species]
    species = [Element(s.strip()) for s in jarvis_atoms.elements]
    coords = jarvis_atoms.frac_coords
    lattice = jarvis_atoms.lattice_mat
    pmg_structure = Structure(lattice, species, 
                              coords, coords_are_cartesian = False)
    return pmg_structure

# %%
