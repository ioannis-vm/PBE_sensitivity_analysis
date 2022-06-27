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

