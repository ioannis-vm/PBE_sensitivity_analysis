"""
Generate taskfiles to run dynamic analyses on SAVIO
"""

import sys
sys.path.append("src")

from util import read_study_param

hazard_lvl_dirs = ['hazard_level_' + str(i+1)
                   for i in range(16)]

gms = ['gm'+str(i+1) for i in range(14)]

archetypes_all = read_study_param('data/archetype_codes_response').split()

# extract unique cases
archetypes = []
for arch in archetypes_all:
    if arch not in archetypes:
        archetypes.append(arch)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for arch in archetypes:
    with open(f'savio/taskfile_{arch}', 'w') as file:
        for hz in hazard_lvl_dirs:
            for gm in gms:
                file.write(f"/global/home/users/ioannisvm/.conda/envs/computing/bin/python ../src/response_{arch}.py '--gm_dir' '../analysis/{arch}/{hz}/ground_motions' '--gm_dt' '0.005' '--analysis_dt' '0.001' '--gm_number' '{gm}' '--output_dir' 'analysis/{arch}/{hz}/response/{gm}'\n")
