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
        self.preamble += text + '\n\n'

    def add_rule(self, target: str, prerequisites: list[str],
                 recipe: list[str], phony=False):
        self.rules.append(Rule(target, prerequisites, recipe, phony))

    def _rule_text(self, rule):
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
        with open(path, 'w') as f:
            f.write(self.preamble)
            for rule in self.rules:
                f.write(self._rule_text(rule))
            phony_rules = []
            for rule in self.rules:
                if rule.phony == True:
                    phony_rules.append(rule)
            ph_rule = Rule(
                '.PHONY',
                [ph.target for ph in phony_rules], [])
            f.write(self._rule_text(ph_rule))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setup #
# ~~~~~ #

hazard_lvl_dirs = ['hazard_level_' + str(i+1)
                   for i in range(16)]

gms = ['gm'+str(i+1) for i in range(14)]

cases = ['office3', 'healthcare3']
response_cases = ['office3']
use_response_of = dict(office3='office3', healthcare3='office3')

rvgroups = ['edp', 'cmp_quant', 'cmp_dm', 'cmp_dv', 'bldg_dm', 'bldg_dv']
uncertainty_cases = ['low', 'medium']
repl_threshold_cases = [0.4, 1.0]

num_levels = 3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mkf = Makefile()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Parsing raw PEER ground motion files #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

prereqs = []
for hz in hazard_lvl_dirs:
    for case in response_cases:
        prereqs.append(f"make/{case}/{hz}/ground_motions/ground_motions_parsed")

mkf.add_rule(
    "make/all_ground_motions_parsed",
    prereqs,
    ["touch make/all_ground_motions_parsed"])

for case in response_cases:
    for hz in hazard_lvl_dirs:
        input_dir = f'analysis/{case}/{hz}/ground_motions/peer_raw'
        output_dir = f'analysis/{case}/{hz}/ground_motions/parsed'
        plot_dir = f'figures/{case}/{hz}/ground_motions'
        mkf.add_rule(
            f"make/{case}/{hz}/ground_motions/ground_motions_parsed",
            [
                f'analysis/{case}/{hz}/ground_motions/peer_raw/_SearchResults.csv',
                'src/parse_gms.py',
                'src/ground_motion_utils.py'],
            [
                f"python src/parse_gms.py '--input_dir' '{input_dir}' '--output_dir' '{output_dir}' '--plot_dir' '{plot_dir}'",
                f"mkdir -p make/{case}/{hz}/ground_motions && touch make/{case}/{hz}/ground_motions/ground_motions_parsed"
            ])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Running NLTH analysis to get the building response #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
prereqs = []
for case in response_cases:
    for hz in hazard_lvl_dirs:
        prereqs.append(
            f"make/{use_response_of[case]}/{hz}/response/all_responses_obtained")

mkf.add_rule(
    "make/all_response_summaries_obtained",
    prereqs,
    ["touch make/all_response_summaries_obtained"])

for case in response_cases:
    for hz in hazard_lvl_dirs:
        mkf.add_rule(
            f"make/{use_response_of[case]}/{hz}/response/all_responses_obtained",
            ['src/response_vectors.py'],
            [f"python src/response_vectors.py '--input_dir' 'analysis/{use_response_of[case]}/{hz}/response' '--output_dir' 'analysis/{use_response_of[case]}/{hz}/response_summary' '--num_levels' '{num_levels}' '--num_inputs' '14' && mkdir -p make/{use_response_of[case]}/{hz}/response && touch make/{use_response_of[case]}/{hz}/response/all_responses_obtained"])

# # response figures

# mkf.add_rule(
#     "make/all_response_TH_figures_obtained",
#     [f"make/{case}/{hz}/response/response_TH_figures_obtained"
#      for hz in hazard_lvl_dirs],
#     ["touch make/all_response_TH_figures_obtained"])

# for case in cases:
#     for hz in hazard_lvl_dirs:
#         mkf.add_rule(
#             f"make/{case}/{hz}/response/response_TH_figures_obtained",
#             [f"figures/{case}/{hz}/response/{gm}/FA.pdf" for gm in gms] +
#             [f"figures/{case}/{hz}/response/{gm}/ID.pdf" for gm in gms] +
#             [f"figures/{case}/{hz}/response/{gm}/FV.pdf" for gm in gms],
#             [f"touch make/{case}/{hz}/response/response_TH_figures_obtained"])
# for case in cases:
#     for hz in hazard_lvl_dirs:
#         for gm in gms:
#             mkf.add_rule(
#                 f"figures/{case}/{hz}/response/{gm}/FA.pdf",
#                 [f"make/{case}/{hz}/response/all_responses_obtained",
#                  "src/response_figures_TH.py"],
#                 [f"mkdir -p figures/{case}/{hz}/response/{gm} && python src/response_figures_TH.py '--input_dir' 'analysis/{case}/{hz}/response/{gm}' '--fig_type' 'FA' '--output_filename' 'figures/{case}/{hz}/response/{gm}/FA.pdf' '--num_levels' '{num_levels}'"])
#             mkf.add_rule(
#                 f"figures/{case}/{hz}/response/{gm}/ID.pdf",
#                 [f"make/{case}/{hz}/response/all_responses_obtained",
#                  "src/response_figures_TH.py"],
#                 [f"mkdir -p figures/{case}/{hz}/response/{gm} && python src/response_figures_TH.py '--input_dir' 'analysis/{case}/{hz}/response/{gm}' '--fig_type' 'ID' '--output_filename' 'figures/{case}/{hz}/response/{gm}/ID.pdf' '--num_levels' '{num_levels}'"])
#             mkf.add_rule(
#                 f"figures/{case}/{hz}/response/{gm}/FV.pdf",
#                 [f"make/{case}/{hz}/response/all_responses_obtained",
#                  "src/response_figures_TH.py"],
#                 [f"mkdir -p figures/{case}/{hz}/response/{gm} && python src/response_figures_TH.py '--input_dir' 'analysis/{case}/{hz}/response/{gm}' '--fig_type' 'FV' '--output_filename' 'figures/{case}/{hz}/response/{gm}/FV.pdf' '--num_levels' '{num_levels}'"])

# prereqs = []
# for case in cases:
#     for hz in hazard_lvl_dirs:
#         prereqs.append(
#             f"make/{case}/{hz}/response/response_figures_obtained")

# mkf.add_rule(
#     "make/all_response_figures_obtained",
#     prereqs,
#     ["touch make/all_response_figures_obtained"])

# for case in cases:
#     for hz in hazard_lvl_dirs:
#         mkf.add_rule(
#             f"make/{case}/{hz}/response/response_figures_obtained",
#             [f"figures/{case}/{hz}/response/PID-1.pdf",
#              f"figures/{case}/{hz}/response/PID-2.pdf",
#              f"figures/{case}/{hz}/response/PFA-1.pdf",
#              f"figures/{case}/{hz}/response/PFA-2.pdf",
#              f"figures/{case}/{hz}/response/PFA_norm-1.pdf",
#              f"figures/{case}/{hz}/response/PFA_norm-2.pdf",
#              f"figures/{case}/{hz}/response/PFV-1.pdf",
#              f"figures/{case}/{hz}/response/PFV-2.pdf"],
#             [f"touch make/{case}/{hz}/response/response_figures_obtained"])

# for case in cases:
#     for hz in hazard_lvl_dirs:
#         for direction in ['1', '2']:
#             mkf.add_rule(
#                 f"figures/{case}/{hz}/response/PID-{direction}.pdf",
#                 [f"src/response_figures.py",
#                  f"make/{case}/{hz}/response/all_responses_obtained"],
#                 [f"python src/response_figures.py '--figure_type' 'PID' '--direction' '{direction}' '--input_dir' 'analysis/{case}/{hz}/response' '--output_dir' 'figures/{case}/{hz}/response' '--num_levels' '{num_levels}'"])
#             mkf.add_rule(
#                 f"figures/{case}/{hz}/response/PFA-{direction}.pdf",
#                 [f"src/response_figures.py",
#                  f"make/{case}/{hz}/response/all_responses_obtained"],
#                 [f"python src/response_figures.py '--figure_type' 'PFA' '--direction' '{direction}' '--input_dir' 'analysis/{case}/{hz}/response' '--output_dir' 'figures/{case}/{hz}/response' '--num_levels' '{num_levels}'"])
#             mkf.add_rule(
#                 f"figures/{case}/{hz}/response/PFA_norm-{direction}.pdf",
#                 [f"src/response_figures.py",
#                  f"make/{case}/{hz}/response/all_responses_obtained"],
#                 [f"python src/response_figures.py '--figure_type' 'PFA_norm' '--direction' '{direction}' '--input_dir' 'analysis/{case}/{hz}/response' '--output_dir' 'figures/{case}/{hz}/response' '--num_levels' '{num_levels}'"])
#             mkf.add_rule(
#                 f"figures/{case}/{hz}/response/PFV-{direction}.pdf",
#                 [f"src/response_figures.py",
#                  f"make/{case}/{hz}/response/all_responses_obtained"],
#                 [f"python src/response_figures.py '--figure_type' 'PFV' '--direction' '{direction}' '--input_dir' 'analysis/{case}/{hz}/response' '--output_dir' 'figures/{case}/{hz}/response' '--num_levels' '{num_levels}'"])

# # merging response figures to a single pdf file

# mkf.add_rule(
#     f"make/response_figures_merged",
#     [f"make/{case}/response_figures_merged" for case in cases],
#     f"touch make/response_figures_merged"
# )

# for case in cases:
#     mkf.add_rule(
#         f"make/{case}/response_figures_merged",
#         [f"figures/{case}/merged/response/PFA-1.pdf",
#          f"figures/{case}/merged/response/PFA-2.pdf",
#          f"figures/{case}/merged/response/PFA_norm-1.pdf",
#          f"figures/{case}/merged/response/PFA_norm-2.pdf",
#          f"figures/{case}/merged/response/PFV-1.pdf",
#          f"figures/{case}/merged/response/PFV-2.pdf",
#          f"figures/{case}/merged/response/PID-1.pdf",
#          f"figures/{case}/merged/response/PID-2.pdf"],
#         [f"touch make/{case}/response_figures_merged"])

#     mkf.add_rule(
#         f"figures/{case}/merged/response/%",
#         [f"make/all_response_figures_obtained"],
#         [f"bash -c \"mkdir -p figures/{case}/merged/response && gs -q -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE=figures/{case}/merged/response/$* -dBATCH $$(find . -name '$*' | sort | tr '\\n' ' ' | sed -r 's/\.\///g')\""])



# ~~~~~~~~~~~~~~~~~~~~~~ #
# Performance Evaluation #
# ~~~~~~~~~~~~~~~~~~~~~~ #

targets = []  # initialize
for case in cases:
    for hz in hazard_lvl_dirs:
        for uncertainty_case in uncertainty_cases:
            for repl in repl_threshold_cases:
                for rvgroup in rvgroups:
                    targets.append(f'make/{case}/{hz}/performance/{uncertainty_case}/{repl}/{rvgroup}/performance_obtained')
targets.extend(['make/all_response_summaries_obtained'])

mkf.add_rule(
    "make/performance",
    targets,
    ['mkdir -p make/performance && touch make/performance']
)

for case in cases:
    for hz in hazard_lvl_dirs:
        for uncertainty_case in uncertainty_cases:
            for repl in repl_threshold_cases:
                for rvgroup in rvgroups:
                    mkf.add_rule(
                        f'make/{case}/{hz}/performance/{uncertainty_case}/{repl}/{rvgroup}/performance_obtained',
                        [
                            f'src/performance_var_sens.py',
                            f'src/performance_data_{case}/input_cmp_quant.csv',
                            f'src/performance_data_{case}/input_fragility.csv',
                            f'src/performance_data_{case}/input_repair_cost.csv',
                            'src/p_58_assessment.py',
                            f"make/{use_response_of[case]}/{hz}/response/all_responses_obtained",
                        ],
                        [f"python src/performance_var_sens.py '--response_path' 'analysis/{use_response_of[case]}/{hz}/response_summary/response.csv' '--modeling_uncertainty_case' '{uncertainty_case}' '--repl_thr' '{repl}' '--rv_group' '{rvgroup}' '--performance_data_path' 'src/performance_data_{case}' '--analysis_output_path' 'analysis/{case}/{hz}/performance/{uncertainty_case}/{repl}/{rvgroup}' && mkdir -p make/{case}/{hz}/performance/{uncertainty_case}/{rvgroup} && touch make/{case}/{hz}/performance/{uncertainty_case}/{rvgroup}/performance_obtained"]
                    )

# # result of fixing edp groups to their mean

# targets = []  # initialize
# for case in cases:
#     for hz in hazard_lvl_dirs:
#         for rvgroup in rvgroups:
#             targets.append(f'make/{case}/{hz}/performance/{rvgroup}/fixed_mean_performance_obtained')

# mkf.add_rule(
#     "make/performance/all_fixed_mean_performance_evals_obtained",
#     targets,
#     ['mkdir -p make/performance && touch make/performance/all_fixed_mean_performance_evals_obtained']
# )

# for case in cases:
#     for hz in hazard_lvl_dirs:
#         for rvgroup in rvgroups:
#             mkf.add_rule(
#                 f'make/{case}/{hz}/performance/{rvgroup}/fixed_mean_performance_obtained',
#                 [
#                     f'src/performance_fix_mean.py',
#                     f'src/performance_data_{case}/input_cmp_quant.csv',
#                     f'src/performance_data_{case}/input_fragility.csv',
#                     f'src/performance_data_{case}/input_repair_cost.csv',
#                     'src/p_58_assessment.py',
#                     f'analysis/{case}/{hz}/response_summary/response.csv',

#                 ],
#                 [f"python src/performance_fix_mean.py '--response_path' 'analysis/{case}/{hz}/response_summary/response.csv' '--rv_group' '{rvgroup}' '--performance_data_path' 'src/performance_data_{case}' '--analysis_output_path' 'analysis/{case}/{hz}/performance/{rvgroup}' '--figures_output_path' 'figures/{case}/{hz}/performance/{rvgroup}' && mkdir -p make/{case}/{hz}/performance/{rvgroup} && touch make/{case}/{hz}/performance/{rvgroup}/fixed_mean_performance_obtained"]
#             )



mkf.write_to_file('Makefile')
