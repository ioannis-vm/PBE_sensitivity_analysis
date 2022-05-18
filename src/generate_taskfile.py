from dataclasses import dataclass, field


hazard_lvl_dirs = ['hazard_level_' + str(i+1)
                   for i in range(16)]

gms = ['gm'+str(i+1) for i in range(14)]

cases = ['office3', 'office10']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for case in cases:
    with open(f'taskfile_{case}', 'w') as file:
        for hz in hazard_lvl_dirs:
            for gm in gms:
                file.write(f"/global/home/users/ioannisvm/.conda/envs/computing/bin/python src/response_{case}.py '--gm_dir' 'analysis/{case}/{hz}/ground_motions/parsed' '--gm_dt' '0.005' '--analysis_dt' '0.001' '--gm_number' '{gm}' '--output_dir' 'analysis/{case}/{hz}/response/{gm}'\n")
