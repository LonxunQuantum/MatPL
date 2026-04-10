import numpy as np

element_table = [
    '', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La',
    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac',
    'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

element_table_2 = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
    'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39,
    'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47,
    'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57,
    'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
    'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79,
    'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
    'Fr': 87, 'Ra': 88, 'Ac': 89,
    'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97,
    'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
    'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
    'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114,
    'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

physical = {
    "atomic_number" : {
    1:"H", 2:"He", 3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 10:"Ne", 
    11:"Na", 12:"Mg", 13:"Al", 14:"Si", 15:"P", 16:"S", 17:"Cl", 18:"Ar", 19:"K", 20:"Ca", 
    21:"Sc", 22:"Ti", 23:"V", 24:"Cr", 25:"Mn", 26:"Fe", 27:"Co", 28:"Ni", 29:"Cu", 30:"Zn",
    31:"Ga", 32:"Ge", 33:"As", 34:"Se", 35:"Br", 36:"Kr", 37:"Rb", 38:"Sr", 39:"Y", 40:"Zr",
    41:"Nb", 42:"Mo", 43:"Tc", 44:"Ru", 45:"Rh", 46:"Pd", 47:"Ag", 48:"Cd", 49:"In", 50:"Sn",
    51:"Sb", 52:"Te", 53:"I", 54:"Xe", 55:"Cs", 56:"Ba", 57:"La", 58:"Ce", 59:"Pr", 60:"Nd", 
    61:"Pm", 62:"Sm", 63:"Eu", 64:"Gd", 65:"Tb", 66:"Dy", 67:"Ho", 68:"Er", 69:"Tm", 70:"Yb", 
    71:"Lu", 72:"Hf", 73:"Ta", 74:"W", 75:"Re", 76:"Os", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg", 
    81:"Tl", 82:"Pb", 83:"Bi", 84:"Po", 85:"At", 86:"Rn", 87:"Fr", 88:"Ra", 89:"Ac", 90:"Th", 
    91:"Pa", 92:"U", 93:"Np", 94:"Pu", 95:"Am", 96:"Cm", 97:"Bk", 98:"Cf", 99:"Es", 100:"Fm", 
    101:"Md", 102:"No", 103:"Lr", 104:"Rf", 105:"Db", 106:"Sg", 107:"Bh", 108:"Hs", 109:"Mt"
    },
    
    "atom_mass" : {
    "H": 1.00794, "He": 4.002602, "Li": 6.941, "Be": 9.012182, "B": 10.811, "C": 12.0107, "N": 14.0067, "O": 15.9994, "F": 18.9984032, 
    "Ne": 20.1797, "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815386, "Si": 28.0855, "P": 30.973762, "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.0983, "Ca": 40.078, 
    "Sc": 44.955912, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938045, "Fe": 55.845, "Co": 58.933195, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38, 
    "Ga": 69.723, "Ge": 72.64, "As": 74.9216, "Se": 78.96, "Br": 79.904, "Kr": 83.798, "Rb": 85.4678, "Sr": 87.62, "Y": 88.90585, "Zr": 91.224, 
    "Nb": 92.90638, "Mo": 95.96, "Tc": 98, "Ru": 101.07, "Rh": 102.9055, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.411, "In": 114.818, "Sn": 118.71, 
    "Sb": 121.76, "Te": 127.6, "I": 126.90447, "Xe": 131.293, "Cs": 132.9054519, "Ba": 137.327, "La": 138.90547, "Ce": 140.116, "Pr": 140.90765, "Nd": 144.242, 
    "Pm": 145, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.92535, "Dy": 162.5, "Ho": 164.93032, "Er": 167.259, "Tm": 168.93421, "Yb": 173.054, 
    "Lu": 174.9668, "Hf": 178.49, "Ta": 180.94788, "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.966569, "Hg": 200.59, 
    "Tl": 204.3833, "Pb": 207.2, "Bi": 208.9804, "Po": 209, "At": 210, "Rn": 222, "Fr": 223, "Ra": 226, "Ac": 227, "Th": 232.03806, 
    "Pa": 231.03588, "U": 238.02891, "Np": 237, "Pu": 244, "Am": 243, "Cm": 247, "Bk": 247, "Cf": 251, "Es": 252, "Fm": 257, 
    "Md": 258, "No": 259, "Lr": 262, "Rf": 267, "Db": 268, "Sg": 271, "Bh": 272, "Hs": 270, "Mt": 276,
    },
    
    "atom_radius" : { 
    "H": 25, "He": float("NaN"), "Li": 145, "Be": 105, "B": 85, "C": 70, "N": 65, "O": 60, "F": 50, 
    "Ne": float("NaN"), "Na": 180, "Mg": 150, "Al": 125, "Si": 110, "P": 100, "S": 100, "Cl": 100, "Ar": float("NaN"), "K": 220, "Ca": 180, 
    "Sc": 160, "Ti": 140, "V": 135, "Cr": 140, "Mn": 140, "Fe": 140, "Co": 135, "Ni": 135, "Cu": 135, "Zn": 135, 
    "Ga": 130, "Ge": 125, "As": 115, "Se": 115, "Br": 115, "Kr": float("NaN"), "Rb": 235, "Sr": 200, "Y": 180, "Zr": 155, 
    "Nb": 145, "Mo": 145, "Tc": 135, "Ru": 130, "Rh": 135, "Pd": 140, "Ag": 160, "Cd": 155, "In": 155, "Sn": 145, 
    "Sb": 145, "Te": 140, "I": 140, "Xe": float("NaN"), "Cs": 260, "Ba": 215, "La": 195, "Ce": 185, "Pr": 185, "Nd": 185, 
    "Pm": 185, "Sm": 185, "Eu": 185, "Gd": 180, "Tb": 175, "Dy": 175, "Ho": 175, "Er": 175, "Tm": 175, "Yb": 175, 
    "Lu": 175, "Hf": 155, "Ta": 145, "W": 135, "Re": 135, "Os": 130, "Ir": 135, "Pt": 135, "Au": 135, "Hg": 150, 
    "Tl": 190, "Pb": 180, "Bi": 160, "Po": 190, "At": float("NaN"), "Rn": float("NaN"), "Fr": float("NaN"), "Ra": 215, "Ac": 195, "Th": 180, 
    "Pa": 180, "U": 175, "Np": 175, "Pu": 175, "Am": 175, "Cm": float("NaN"), "Bk": float("NaN"), "Cf": float("NaN"), "Es": float("NaN"), "Fm": float("NaN"), 
    "Md": float("NaN"), "No": float("NaN"), "Lr": float("NaN"), "Rf": float("NaN"), "Db": float("NaN"), "Sg": float("NaN"), "Bh": float("NaN"), "Hs": float("NaN"), "Mt": float("NaN"),
    },
    "molar_vol" : {
    "H": 11.42, "He": 21, "Li": 13.02, "Be": 4.85, "B": 4.39, "C": 5.29, "N": 13.54, "O": 17.36, "F": 11.2, 
    "Ne": 13.23, "Na": 23.78, "Mg": 14, "Al": 10, "Si": 12.06, "P": 17.02, "S": 15.53, "Cl": 17.39, "Ar": 22.56, "K": 45.94, "Ca": 26.2, 
    "Sc": 15, "Ti": 10.64, "V": 8.32, "Cr": 7.23, "Mn": 7.35, "Fe": 7.09, "Co": 6.67, "Ni": 6.59, "Cu": 7.11, "Zn": 9.16, 
    "Ga": 11.8, "Ge": 13.63, "As": 12.95, "Se": 16.42, "Br": 19.78, "Kr": 27.99, "Rb": 55.76, "Sr": 33.94, "Y": 19.88, "Zr": 14.02, 
    "Nb": 10.83, "Mo": 9.38, "Tc": 8.63, "Ru": 8.17, "Rh": 8.28, "Pd": 8.56, "Ag": 10.27, "Cd": 13, "In": 15.76, "Sn": 16.29, 
    "Sb": 18.19, "Te": 20.46, "I": 25.72, "Xe": 35.92, "Cs": 70.94, "Ba": 38.16, "La": 22.39, "Ce": 20.69, "Pr": 20.8, "Nd": 20.59, 
    "Pm": 20.23, "Sm": 19.98, "Eu": 28.97, "Gd": 19.9, "Tb": 19.3, "Dy": 19.01, "Ho": 18.74, "Er": 18.46, "Tm": 19.1, "Yb": 24.84, 
    "Lu": 17.78, "Hf": 13.44, "Ta": 10.85, "W": 9.47, "Re": 8.86, "Os": 8.42, "Ir": 8.52, "Pt": 9.09, "Au": 10.21, "Hg": 14.09, 
    "Tl": 17.22, "Pb": 18.26, "Bi": 21.31, "Po": 22.97, "At": float("NaN"), "Rn": 50.5, "Fr": float("NaN"), "Ra": 41.09, "Ac": 22.55, "Th": 19.8, 
    "Pa": 15.18, "U": 12.49, "Np": 11.59, "Pu": 12.29, "Am": 17.63, "Cm": 18.05, "Bk": 16.84, "Cf": 16.5, "Es": 28.52, "Fm": float("NaN"), 
    "Md": float("NaN"), "No": float("NaN"), "Lr": float("NaN"), "Rf": float("NaN"), "Db": float("NaN"), "Sg": float("NaN"), "Bh": float("NaN"), "Hs": float("NaN"), "Mt": float("NaN"),
    },
    "_molar_vol":"molar volume (cm^3/mol)",
    "melting_point" : {
    "H": 14.01, "He": 0.95, "Li": 453.69, "Be": 1560, "B": 2349, "C": 3800, "N": 63.05, "O": 54.8, "F": 53.53, 
    "Ne": 24.56, "Na": 370.87, "Mg": 923, "Al": 933.47, "Si": 1687, "P": 317.3, "S": 388.36, "Cl": 171.6, "Ar": 83.8, "K": 336.53, "Ca": 1115, 
    "Sc": 1814, "Ti": 1941, "V": 2183, "Cr": 2180, "Mn": 1519, "Fe": 1811, "Co": 1768, "Ni": 1728, "Cu": 1357.77, "Zn": 692.68, 
    "Ga": 302.91, "Ge": 1211.4, "As": 1090, "Se": 494, "Br": 265.8, "Kr": 115.79, "Rb": 312.46, "Sr": 1050, "Y": 1799, "Zr": 2128, 
    "Nb": 2750, "Mo": 2896, "Tc": 2430, "Ru": 2607, "Rh": 2237, "Pd": 1828.05, "Ag": 1234.93, "Cd": 594.22, "In": 429.75, "Sn": 505.08, 
    "Sb": 903.78, "Te": 722.66, "I": 386.85, "Xe": 161.4, "Cs": 301.59, "Ba": 1000, "La": 1193, "Ce": 1068, "Pr": 1208, "Nd": 1297, 
    "Pm": 1373, "Sm": 1345, "Eu": 1099, "Gd": 1585, "Tb": 1629, "Dy": 1680, "Ho": 1734, "Er": 1802, "Tm": 1818, "Yb": 1097, 
    "Lu": 1925, "Hf": 2506, "Ta": 3290, "W": 3695, "Re": 3459, "Os": 3306, "Ir": 2739, "Pt": 2041.4, "Au": 1337.33, "Hg": 234.32, 
    "Tl": 577, "Pb": 600.61, "Bi": 544.4, "Po": 527, "At": 575, "Rn": 202, "Fr": 300, "Ra": 973,
    },
    "boiling_point" : {
    "H": 20.28, "He": 4.22, "Li": 1615, "Be": 2742, "B": 4200, "C": 4300, "N": 77.36, "O": 90.2, "F": 85.03, 
    "Ne": 27.07, "Na": 1156, "Mg": 1363, "Al": 2792, "Si": 3173, "P": 550, "S": 717.87, "Cl": 239.11, "Ar": 87.3, "K": 1032, "Ca": 1757, 
    "Sc": 3103, "Ti": 3560, "V": 3680, "Cr": 2944, "Mn": 2334, "Fe": 3134, "Co": 3200, "Ni": 3186, "Cu": 3200, "Zn": 1180, 
    "Ga": 2477, "Ge": 3093, "As": 887, "Se": 958, "Br": 332, "Kr": 119.93, "Rb": 961, "Sr": 1655, "Y": 3609, "Zr": 4682, 
    "Nb": 5017, "Mo": 4912, "Tc": 4538, "Ru": 4423, "Rh": 3968, "Pd": 3236, "Ag": 2435, "Cd": 1040, "In": 2345, "Sn": 2875, 
    "Sb": 1860, "Te": 1261, "I": 457.4, "Xe": 165.1, "Cs": 944, "Ba": 2143, "La": 3743, "Ce": 3633, "Pr": 3563, "Nd": 3373, 
    "Pm": 3273, "Sm": 2076, "Eu": 1800, "Gd": 3523, "Tb": 3503, "Dy": 2840, "Ho": 2993, "Er": 3141, "Tm": 2223, "Yb": 1469, 
    "Lu": 3675, "Hf": 4876, "Ta": 5731, "W": 5828, "Re": 5869, "Os": 5285, "Ir": 4701, "Pt": 4098, "Au": 3129, "Hg": 629.88, 
    "Tl": 1746, "Pb": 2022, "Bi": 1837, "Po": 1235, "At": float("NaN"), "Rn": 211.3, "Fr": float("NaN"), "Ra": 2010,
    },
    "electron_affin" : {
    "H": 72800, "He": 0, "Li": 59600, "Be": 0, "B": 26700, "C": 153900, "N": 7000, "O": 141000, "F": 328000, 
    "Ne": 0, "Na": 52800, "Mg": 0, "Al": 42500, "Si": 133600, "P": 72000, "S": 200000, "Cl": 349000, "Ar": 0, "K": 48400, "Ca": 2370, 
    "Sc": 18100, "Ti": 7600, "V": 50600, "Cr": 64300, "Mn": 0, "Fe": 15700, "Co": 63700, "Ni": 112000, "Cu": 118400, "Zn": 0, 
    "Ga": 28900, "Ge": 119000, "As": 78000, "Se": 195000, "Br": 324600, "Kr": 0, "Rb": 46900, "Sr": 5030, "Y": 29600, "Zr": 41100, 
    "Nb": 86100, "Mo": 71900, "Tc": 53000, "Ru": 101300, "Rh": 109700, "Pd": 53700, "Ag": 125600, "Cd": 0, "In": 28900, "Sn": 107300, 
    "Sb": 103200, "Te": 190200, "I": 295200, "Xe": 0, "Cs": 45500, "Ba": 13950, "La": 48000, "Ce": 50000, "Pr": 50000, "Nd": 50000, 
    "Pm": 50000, "Sm": 50000, "Eu": 50000, "Gd": 50000, "Tb": 50000, "Dy": 50000, "Ho": 50000, "Er": 50000, "Tm": 50000, "Yb": 50000, 
    "Lu": 33000, "Hf": 0, "Ta": 31000, "W": 78600, "Re": 14500, "Os": 106100, "Ir": 151000, "Pt": 205300, "Au": 222800, "Hg": 0, 
    "Tl": 19200, "Pb": 35100, "Bi": 91200, "Po": 183300, "At": 270100, "Rn": float("NaN"),
    },
    "_electron_affin":"# electron affinity (J/mol)", 
    "pauling" : {
    "H": 2.2, "He": float("NaN"), "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "Ne": float("NaN"), "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.9, "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": float("NaN"), "K": 0.82, "Ca": 1, 
    "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.9, "Zn": 1.65, 
    "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, 
    "Nb": 1.6, "Mo": 2.16, "Tc": 1.9, "Ru": 2.2, "Rh": 2.28, "Pd": 2.2, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, 
    "Sb": 2.05, "Te": 2.1, "I": 2.66, "Xe": 2.6, "Cs": 0.79, "Ba": 0.89, "La": 1.1, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, 
    "Pm": float("NaN"), "Sm": 1.17, "Eu": float("NaN"), "Gd": 1.2, "Tb": float("NaN"), "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": float("NaN"), 
    "Lu": 1.27, "Hf": 1.3, "Ta": 1.5, "W": 2.36, "Re": 1.9, "Os": 2.2, "Ir": 2.2, "Pt": 2.28, "Au": 2.54, "Hg": 2, 
    "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Po": 2, "At": 2.2, "Rn": float("NaN"), "Fr": 0.7, "Ra": 0.9,
    }, 
    "_pauling":"Pauling electronegativity"
}

physical_scale = {}
for typ in physical.keys():
    if isinstance(physical[typ], dict) is False:
        continue
    if typ == "atomic_number":
        data = np.array(list(physical[typ].keys()))
    else:
        data = np.array(list(physical[typ].values()))
        data = data[~np.isnan(data)]
    min_value = np.min(data)
    max_value = np.max(data)
    physical_scale[typ] = [min_value, max_value]

def get_normalized_data(atom_type, type_list):
    dicts = {}
    atom_name = physical["atomic_number"][atom_type]
    # atomic_number
    for typ in type_list:
        value = atom_type if typ == "atomic_number" else physical[typ][atom_name]
        if np.isnan(value):
            raise Exception("the physical property {} of atom {} in dict is NaN !".format(typ, atom_name))
        normalized_data = (value - physical_scale[typ][0]) / (physical_scale[typ][1] - physical_scale[typ][0])
        dicts[typ] = normalized_data
    return dicts

def get_normalized_data_list(type_map, type_list):
    dicts = {}
    for atom in type_map:
        if atom == 0:
            continue
        dicts[atom] = list(get_normalized_data(atom, type_list).values())
    return dicts

def get_atomic_number_from_name(atomic_names:list[str]):
    res = []
    for name in atomic_names:
        res.append(element_table_2[name])
    return res

if __name__ == "__main__":
    # keys "atomic_number", "atom_mass", "atom_radius", "molar_vol", "melting_point", "boiling_point", "electron_affin", "pauling"
    res = get_normalized_data_list([3,29], ["atomic_number", "atom_mass", "atom_radius", "molar_vol", "melting_point", "boiling_point", "electron_affin", "pauling"])
    print(res)

