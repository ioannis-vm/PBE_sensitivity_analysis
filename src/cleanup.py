import os
import shutil

archetype_codes = [
    'smrf_3_of_II',
    'smrf_6_of_II',
    'smrf_9_of_II',
    'smrf_3_he_II'
    'smrf_6_he_II'
    'smrf_9_he_II'
    'smrf_3_of_IV',
    'smrf_6_of_IV',
    'smrf_3_he_IV',
    'smrf_6_he_IV',
]
hazard_level_dirs = [f'hazard_level_{i+1}' for i in range(16)]

# clean up analysis directory
dirs_to_remove = ['performance', 'response']

for arch in archetype_codes:
    for hz in hazard_level_dirs:
        for dr in dirs_to_remove:
            path = f'analysis/{arch}/{hz}/{dr}'
            if os.path.isdir(path):
                shutil.rmtree(path)
