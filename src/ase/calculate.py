import numpy as np
import torch
from utils.file_operation import check_model_type
from src.mods.infer import Inference

from ase.calculators.calculator import (
    Calculator,
    all_changes,
    PropertyNotImplementedError,
)

class MatPL_calculator(Calculator):

    """ASE calculator
    supported properties: Etot, Ei, forces, virial

    >>> calc = MatPL(model_file='ckptfile or nep txt file')
    >>> atoms = bulk('C', 'diamond', cubic=True)
    >>> atoms.calc = calc
    >>> energy = atoms.get_potential_energy()
    >>> forces = atoms.get_forces()
    >>> stress = atoms.get_stress()

    """

    implemented_properties = [
        "energy",
        "energies",
        "forces",
        "stress"
    ]

    def __init__(self, model_file=None, **kwargs) -> None:
        """Initialize calculator

        Args:
            model_file (str, optional): filename of nep model. Defaults to "nep.txt".
        """
        Calculator.__init__(self, **kwargs)
        # load model cpkts or txt
        model_type = check_model_type(model_load_path=model_file)
        if model_type not in ["DP", "NEP"]:
            raise Exception("Error! The input model type is {}! Only support DP or NEP model!".format(model_type))
        self.model_type = model_type
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.calc = Inference(model_file, self.device)

    def __repr__(self):
        ret = "The model type is {}".format(self.model_type)
        return ret

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        # dp
        if self.model_type == "DP":
            Etot, Ei, Force, Virial = self.calc.ase_dp_infer(lattice = np.array(atoms.cell),
                        frac_postions = atoms.get_scaled_positions(),
                        symbols = self.atoms.get_chemical_symbols()
                        )
        if self.model_type == "NEP":
            Etot, Ei, Force, Virial = self.calc.ase_nep_infer(lattice = np.array(atoms.cell),
                        cart_postions = atoms.get_positions(),
                        symbols = self.atoms.get_chemical_symbols()
                        )

        self.results["energy"]   = Etot
        self.results["energies"] = Ei
        self.results["forces"] = Force
        stress = -Virial / atoms.get_volume()
        self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]

class JointCalculator(Calculator):
    implemented_properties = [
        "energy",
        "forces",
        "stress",
    ]

    def __init__(self, *args, **kwargs) -> None:
        Calculator.__init__(self, **kwargs)
        self.calc_list = args

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))
        self.results["stress"] = np.zeros(6)
        for calc in self.calc_list:
            self.results["energy"] += calc.get_potential_energy(self.atoms)
            self.results["forces"] += calc.get_forces(self.atoms)
            self.results["stress"] += calc.get_stress(self.atoms)