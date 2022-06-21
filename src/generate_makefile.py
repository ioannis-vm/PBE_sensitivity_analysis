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

gms = ['gm'+str(i+1) for i in range(14)]

cases = ['smrf_3_of_II']

response_cases = ['smrf_3_of_II']
use_response_of = dict(smrf_3_of_II='smrf_3_of_II')

rvgroups = ['edp', 'cmp_quant', 'cmp_dm', 'cmp_dv', 'bldg_dm', 'bldg_dv']
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
    ["Makefile", "src/site_hazard.py"],
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
    ["Makefile", "src/GMMCalc.java", "src/DisaggregationCalc.java", "src/site_hazard_deagg.sh"],
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
    ["Makefile", "src/site_target_spectra.py"],
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
    ["Makefile", "src/site_gm_selection.py", "make/site_hazard/flatfile_obtained"],
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



mkf.write_to_file('Makefile')
