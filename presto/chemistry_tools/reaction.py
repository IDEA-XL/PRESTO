# modified from https://github.com/rxn4chemistry/rxn-chemutils

from typing import List, Optional, Iterable, Iterator, Generator

def multicomponent_smiles_to_list(
    multicomponent_smiles: str, fragment_bond: Optional[str] = None
) -> List[str]:
    """
    Convert a string of molecules into a list of molecules (taking fragment bonds into account).

    Args:
        multicomponent_smiles: multicomponent SMILES string to convert to a list.
        fragment_bond: fragment bond.

    Returns:
        The list of molecule SMILES comprised in the multi-component SMILES string.
    """
    molecules = multicomponent_smiles.split(".")
    molecules = [molecule for molecule in molecules if molecule != ""]

    # replace fragment bonds if necessary
    if fragment_bond is not None:
        molecules = [molecule.replace(fragment_bond, ".") for molecule in molecules]
    return molecules


def list_to_multicomponent_smiles(
    molecules: Iterable[str], fragment_bond: Optional[str] = None
) -> str:
    """
    Convert a list of molecules into a string representation (taking fragment
    bonds into account).

    Args:
        molecules: molecule SMILES strings to merge into a multi-component SMILES string.
        fragment_bond: fragment bond.

    Returns:
        A multi-component SMILES string.
    """
    # replace fragment bonds if necessary
    if fragment_bond is not None:
        molecules = [molecule.replace(".", fragment_bond) for molecule in molecules]

    return ".".join(molecules)


class ReactionEquation:
    """
    Defines a reaction equation, as given by the molecules involved in a reaction.

    Attributes:
        reactants: SMILES strings for compounds on the left of the reaction arrow.
        agents: SMILES strings for compounds above the reaction arrow. Are
            sometimes merged with the reactants.
        products: SMILES strings for compounds on the right of the reaction arrow.
    """

    reactants: List[str]
    agents: List[str]
    products: List[str]

    def __init__(
        self, reactants: Iterable[str], agents: Iterable[str], products: Iterable[str]
    ):
        """Overwrite init function in order to enable instantiation from any iterator and
        to force copying the lists.
        """
        self.__attrs_init__(list(reactants), list(agents), list(products))

    def __iter__(self) -> Iterator[List[str]]:
        """Helper function to simplify functionality acting on all three
        compound groups"""
        return (i for i in (self.reactants, self.agents, self.products))

    def iter_all_smiles(self) -> Generator[str, None, None]:
        """Helper function to iterate over all the SMILES in the reaction equation"""
        return (molecule for group in self for molecule in group)

    def to_string(self, fragment_bond: Optional[str] = None) -> str:
        """Convert a ReactionEquation to an "rxn" reaction SMILES.
        """
        smiles_groups = (
            list_to_multicomponent_smiles(group, fragment_bond) for group in self
        )
        return ">>".join(smiles_groups)

    @classmethod
    def from_string(
        cls, reaction_string: str, fragment_bond: Optional[str] = None
    ) -> "ReactionEquation":
        """Convert a ReactionEquation from an "rxn" reaction SMILES.
        """

        groups = [
            multicomponent_smiles_to_list(smiles_group, fragment_bond=fragment_bond)
            for smiles_group in reaction_string.split(">>")
        ]

        try:
            return cls(*groups)
        except TypeError as e:
            raise ValueError(
                f"Could not convert {reaction_string} to a ReactionEquation"
            ) from e