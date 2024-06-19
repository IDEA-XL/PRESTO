from bioagent.modalities.molecule_2d import Molecule2DModality

MODALITY_BUILDERS = {
    "molecule_2d": lambda : [Molecule2DModality()],
}
