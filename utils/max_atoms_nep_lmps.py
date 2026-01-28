

def calcualte_atomnum_lmps():
    memory_limit_mb = 40*1024  # 显存大小 24 GB 转换为 MB
    atom_nums = 1

    while True:
        memory_needed_mb = 0
        atom_nums += 1

        # Calculate memory needed
        memory_needed_mb += atom_nums * 4 / 1024 / 1024  # NN_radial
        memory_needed_mb += atom_nums * 500 * 4 / 1024 / 1024  # NL_radial
        memory_needed_mb += atom_nums * 4 / 1024 / 1024  # NN_angular
        memory_needed_mb += atom_nums * 100 * 4 / 1024 / 1024  # NL_angular
        memory_needed_mb += atom_nums * 4 / 1024 / 1024  # potential_per_atom
        memory_needed_mb += atom_nums * 4 / 1024 / 1024  # itype
        memory_needed_mb += atom_nums * 50 * 4 / 1024 / 1024  # Fp
        memory_needed_mb += atom_nums * 15 * 24 * 4 / 1024 / 1024  # sum_fxyz
        memory_needed_mb += atom_nums * 1 * 3 * 4 / 1024 / 1024  # force_per_atom
        memory_needed_mb += atom_nums * 1 * 9 * 4 / 1024 / 1024  # virial_per_atom
        memory_needed_mb += atom_nums * 1 * 3 * 4 / 1024 / 1024  # position

        # Check if memory requirement exceeds the limit
        if memory_needed_mb > memory_limit_mb:
            break

    print(f"Maximum atom_nums: {atom_nums - 1}")
    print(f"Memory needed (MB): {memory_needed_mb}")
    return atom_nums

def need_memroy():
    atom_nums = 2592000
    memory_needed_mb = 0
    atom_nums += 1

    # Calculate memory needed
    memory_needed_mb += atom_nums * 4 / 1024 / 1024  # NN_radial
    memory_needed_mb += atom_nums * 500 * 4 / 1024 / 1024  # NL_radial
    memory_needed_mb += atom_nums * 4 / 1024 / 1024  # NN_angular
    memory_needed_mb += atom_nums * 100 * 4 / 1024 / 1024  # NL_angular
    memory_needed_mb += atom_nums * 8 / 1024 / 1024  # potential_per_atom
    memory_needed_mb += atom_nums * 4 / 1024 / 1024  # ilist
    memory_needed_mb += atom_nums * 4 / 1024 / 1024  # numneigh
    memory_needed_mb += atom_nums * 500 * 4 / 1024 / 1024  # firstneigh
    memory_needed_mb += atom_nums * 500 * 6 * 4 / 1024 / 1024  # r12
    memory_needed_mb += atom_nums * 50 * 4 / 1024 / 1024  # Fp
    memory_needed_mb += atom_nums * 15 * 24 * 4 / 1024 / 1024  # sum_fxyz
    memory_needed_mb += atom_nums * 1.5 * 3 * 8 / 1024 / 1024  # force_per_atom
    memory_needed_mb += atom_nums * 1.5 * 9 * 8 / 1024 / 1024  # virial_per_atom
    memory_needed_mb += atom_nums * 1.5 * 4 / 1024 / 1024  # type
    memory_needed_mb += atom_nums * 1.5 * 3 * 8 / 1024 / 1024  # position
    print(f"Maximum atom_nums: {atom_nums - 1}")
    print(f"Memory needed (GB): {memory_needed_mb/1024}")    

def calculate_percentage_lmps():
    atom_nums = 1
    res = {}
    res['NN_radial'] = atom_nums * 4   # NN_radial
    res['NL_radial'] = atom_nums * 500 * 4   # NL_radial
    res['NN_angular'] = atom_nums * 4   # NN_angular
    res['NL_angular'] = atom_nums * 100 * 4   # NL_angular
    res['potential_per_atom'] = atom_nums * 8   # potential_per_atom
    res['ilist'] = atom_nums * 4   # ilist
    res['numneigh'] = atom_nums * 4   # numneigh
    res['firstneigh'] = atom_nums * 500 * 4   # firstneigh
    res['r12'] = atom_nums * 500 * 6 * 4   # r12
    res['Fp'] = atom_nums * 50 * 4   # Fp
    res['sum_fxyz'] = atom_nums * 15 * 24 * 4   # sum_fxyz
    res['force_per_atom'] = atom_nums * 1.5 * 3 * 8   # force_per_atom
    res['virial_per_atom'] = atom_nums * 1.5 * 9 * 8   # virial_per_atom
    res['type'] = atom_nums * 1.5 * 4   # type
    res['position'] = atom_nums * 1.5 * 3 * 8   # position
    
    all = 0
    for key in res.keys():
        all += res[key]
    
    for key in res.keys():
        print("{}\t{}".format(key, round(res[key]/all*100, 2)))

if __name__ == '__main__':
    # calculate_percentage_lmps()
    # need_memroy()
    calcualte_atomnum_lmps()