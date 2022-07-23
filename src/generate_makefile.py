from dataclasses import dataclass, field


@dataclass
class Rule:
    target: str = field()
    prerequisites: list[str] = field(default_factory=list)
    recipe: list[str] = field(default_factory=list)
    phony: bool = field(default=False)


@dataclass
class Makefile:
    preamble: str = field(default_factory=str)
    rules: list[Rule] = field(default_factory=list)

    def add_preamble(self, text: str):
        """
        Adds the preamble text.
        """
        self.preamble += text + '\n\n'

    def add_rule(self, target: str, prerequisites: list[str],
                 recipe: list[str], phony=False):
        """
        Adds a rule.
        """
        self.rules.append(Rule(target, prerequisites, recipe, phony))

    def _rule_text(self, rule):
        """
        Generates a string containing a rule.
        """
        target = rule.target
        prerequisites = rule.prerequisites
        recipe = rule.recipe
        if prerequisites:
            rule_text = target + ':\\\n'
        else:
            rule_text = target + ':\n'
        for i, p in enumerate(prerequisites):
            if i != len(prerequisites) - 1:
                rule_text += p + '\\\n'
            else:
                rule_text += p + '\n'
        for r in recipe:
            rule_text += '\t' + r + '\n'
        rule_text += '\n'
        return rule_text

    def write_to_file(self, path):
        """
        Generates the Makefile.
        """
        with open(path, 'w') as f:
            f.write(self.preamble)
            for rule in self.rules:
                f.write(self._rule_text(rule))
            phony_rules = []
            for rule in self.rules:
                if rule.phony is True:
                    phony_rules.append(rule)
            ph_rule = Rule(
                '.PHONY',
                [ph.target for ph in phony_rules], [])
            f.write(self._rule_text(ph_rule))


# ~~~~~ #
# Setup #
# ~~~~~ #

hazard_lvl_dirs = ['hazard_level_' + str(i+1)
                   for i in range(16)]

num_gms = 14
gms = ['gm'+str(i+1) for i in range(num_gms)]

# archetype information
cases = [
    'smrf_3_of_II', 'smrf_3_he_II',
    'smrf_6_of_II', 'smrf_6_he_II',
    'smrf_9_of_II', 'smrf_9_he_II',
    'smrf_3_of_IV', 'smrf_3_he_IV',
    'smrf_6_of_IV', 'smrf_6_he_IV'
]
use_response_of = dict(
    smrf_3_of_II='smrf_3_of_II', smrf_3_he_II='smrf_3_of_II',
    smrf_6_of_II='smrf_6_of_II', smrf_6_he_II='smrf_6_of_II',
    smrf_9_of_II='smrf_9_of_II', smrf_9_he_II='smrf_9_of_II',
    smrf_3_of_IV='smrf_3_of_IV', smrf_3_he_IV='smrf_3_of_IV',
    smrf_6_of_IV='smrf_6_of_IV', smrf_6_he_IV='smrf_6_of_IV'
)
num_levels = dict(
    smrf_3_of_II=3,
    smrf_6_of_II=6,
    smrf_9_of_II=9,
    smrf_3_of_IV=3,
    smrf_6_of_IV=6
)

periods = dict(
    smrf_3_of_II=0.93, smrf_6_of_II=1.34, smrf_9_of_II=1.40,
    smrf_3_of_IV=0.77, smrf_6_of_IV=1.07
)
yield_drifts = dict(
    smrf_3_of_II=0.01,
    smrf_6_of_II=0.01,
    smrf_9_of_II=0.01,
    smrf_3_of_IV=0.01,
    smrf_6_of_IV=0.01
)

uncertainty_cases = ['low', 'medium']
repl_threshold_cases = [0.4, 1.0]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mkf = Makefile()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Update makefile when this file changes #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

mkf.add_rule(
    "Makefile",
    ["src/generate_makefile.py"],
    ["python src/generate_makefile.py"]
)


# ~~~~~~~ #
# Phase A #
# ~~~~~~~ #

mkf.add_rule(
    "make/phase_A",
    ["make/site_hazard/hazard_curves_obtained", "make/site_hazard/hazard_levels_defined", "make/site_hazard/deaggregation_complete", "make/site_hazard/target_spectra", "make/site_hazard/ground_motions_selected"],
    ["touch make/phase_A"]
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Generating site hazard curves #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

mkf.add_rule(
    "make/site_hazard/hazard_curves_obtained",
    ["Makefile", "src/HazardCurveCalc.java", "src/site_hazard_curves.sh"],
    ["./src/site_hazard_curves.sh && mkdir -p make/site_hazard && touch make/site_hazard/hazard_curves_obtained"]
)

mkf.add_rule(
    "make/site_hazard/UPD_hazard_curves_obtained",
    [],
    ["mkdir -p make/site_hazard && touch make/site_hazard/hazard_curves_obtained"]
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Obtain UHS and hazard levels #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

mkf.add_rule(
    "make/site_hazard/hazard_levels_defined",
    ["Makefile", "src/site_hazard.py", "make/site_hazard/hazard_curves_obtained"],
    ["python src/site_hazard.py && mkdir -p make/site_hazard && touch make/site_hazard/hazard_levels_defined"]
)

mkf.add_rule(
    "make/site_hazard/UPD_hazard_levels_defined",
    [],
    ["mkdir -p make/site_hazard && touch make/site_hazard/hazard_levels_defined"]
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Seismic deaggregation and GMM values #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

mkf.add_rule(
    "make/site_hazard/deaggregation_complete",
    ["Makefile", "src/GMMCalc.java", "src/DisaggregationCalc.java", "src/site_hazard_deagg.sh", "make/site_hazard/hazard_levels_defined"],
    ["./src/site_hazard_deagg.sh && mkdir -p make/site_hazard && touch make/site_hazard/deaggregation_complete"]
)

mkf.add_rule(
    "make/site_hazard/UPD_deaggregation_complete",
    [],
    ["mkdir -p make/site_hazard && touch make/site_hazard/deaggregation_complete"]
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Target spectra generation #
# ~~~~~~~~~~~~~~~~~~~~~~~~~ #

mkf.add_rule(
    "make/site_hazard/target_spectra",
    ["Makefile", "src/site_target_spectra.py", "make/site_hazard/deaggregation_complete"],
    ["python src/site_target_spectra.py && mkdir -p make/site_hazard && touch make/site_hazard/target_spectra"]
)

mkf.add_rule(
    "make/site_hazard/UPD_target_spectra",
    [],
    ["mkdir -p make/site_hazard && touch make/site_hazard/target_spectra"]
)

# ~~~~~~~~~~~~~~~~~~~~~~~ #
# Ground Motion Selection #
# ~~~~~~~~~~~~~~~~~~~~~~~ #

mkf.add_rule(
    "make/site_hazard/ground_motions_selected",
    ["Makefile", "src/site_gm_selection.py", "make/site_hazard/flatfile_obtained", "make/site_hazard/target_spectra"],
    ["python src/site_gm_selection.py && touch make/site_hazard/ground_motions_selected"]
)

mkf.add_rule(
    "make/site_hazard/UPD_ground_motions_selected",
    [],
    ["touch make/site_hazard/ground_motions_selected"]
)

mkf.add_rule(
    "make/site_hazard/flatfile_obtained",
    ["Makefile", "src/site_gm_selection.sh", "src/convert_xls_csv.py"],
    ["./src/site_gm_selection.sh && touch make/site_hazard/flatfile_obtained"]
)

mkf.add_rule(
    "make/site_hazard/UPD_flatfile_obtained",
    [],
    ["touch make/site_hazard/flatfile_obtained"]
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# at this point, download ground motions from the PEER website

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~ #
# Phase B #
# ~~~~~~~ #

mkf.add_rule(
    "make/phase_B",
    ["make/site_hazard/ground_motions_parsed"],
    ["touch make/phase_B"]
)

# ~~~~~~~~~~~~~~~~~~~~~ #
# Ground Motion Parsing #
# ~~~~~~~~~~~~~~~~~~~~~ #

mkf.add_rule(
    "make/site_hazard/ground_motions_parsed",
    ["Makefile", "src/parse_gms.py", "make/site_hazard/ground_motions_selected"],
    ["python src/parse_gms.py && touch make/site_hazard/ground_motions_parsed"]
)

mkf.add_rule(
    "make/site_hazard/UPD_ground_motions_parsed",
    [],
    ["touch make/site_hazard/ground_motions_parsed"]
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# at this point, run nlth analysis on savio

response_archetypes = []
for arch in cases:
    resp_arch = use_response_of[arch]
    if resp_arch not in response_archetypes:
        response_archetypes.append(resp_arch)

for arch in response_archetypes:
    with open(f'savio/nlth_taskfile_{arch}', 'w') as file:
        for hz in hazard_lvl_dirs:
            for gm in gms:
                file.write(f"/global/home/users/ioannisvm/.conda/envs/computing/bin/python src/response.py '--archetype' '{arch}' '--gm_dir' 'analysis/{arch}/{hz}/ground_motions' '--gm_dt' '0.005' '--analysis_dt' '0.001' '--gm_number' '{gm}' '--output_dir' 'analysis/{arch}/{hz}/response/{gm}'\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~ #
# Phase C #
# ~~~~~~~ #

mkf.add_rule(
    "make/phase_C",
    ["make/response_parsed"],
    ["touch make/phase_C"]
)

# ~~~~~~~~~~~~~~~~~ #
# Parse nlth output #
# ~~~~~~~~~~~~~~~~~ #

prerequisites = []
for case in response_archetypes:
    for hz in hazard_lvl_dirs:
        resp_case = use_response_of[case]
        prerequisites.append(f"make/{resp_case}/{hz}/response/response_parsed")

mkf.add_rule(
    "make/response_parsed",
    prerequisites,
    ["touch make/response_parsed"]
)

for resp_case in response_archetypes:
    for hz in hazard_lvl_dirs:
        nlvl = num_levels[resp_case]
        t_1 = periods[resp_case]
        dry = yield_drifts[resp_case]
        mkf.add_rule(
            f"make/{resp_case}/{hz}/response/response_parsed",
            ["Makefile", "src/response_vectors.py"],
            [f"python src/response_vectors.py '--input_dir' 'analysis/{resp_case}/{hz}/response' '--output_dir' 'analysis/{resp_case}/{hz}/response_summary' '--num_levels' '{nlvl}' '--num_inputs' '{num_gms}' '--t_1' '{t_1}' '--yield_dr' '{dry}' && mkdir -p make/{resp_case}/{hz}/response && touch make/{resp_case}/{hz}/response/response_parsed"]
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# at this point, run sensitivity analysis on savio

for case in cases:
    with open(f'savio/si_taskfile_{case}', 'w') as file:
        for mdl_unc in uncertainty_cases:
            for repl in repl_threshold_cases:
                for hz in hazard_lvl_dirs:
                    file.write(f"/global/home/users/ioannisvm/.conda/envs/computing/bin/python src/performance_var_sens.py '--response_path' 'analysis/{use_response_of[case]}/{hz}/response_summary/response.csv' '--base_period' '{periods[use_response_of[case]]}' '--modeling_uncertainty_case' '{mdl_unc}' '--repl_thr' '{repl}' '--performance_data_path' 'data/{case}/performance' '--analysis_output_path' 'analysis/{case}/{hz}/performance'\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


mkf.write_to_file('Makefile')
