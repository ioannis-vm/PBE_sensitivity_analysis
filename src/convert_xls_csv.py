"""
Converts a given xls file to a csv file, using the Pandas defaults.
"""

import pandas as pd
import argparse

# ~~~~~~~~~~~~~~~~~~~~~~ #
# set up argument parser #
# ~~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--xls_file')
parser.add_argument('--csv_file')
args = parser.parse_args()

df = pd.read_excel(args.xls_file)
df.to_csv(args.csv_file)
