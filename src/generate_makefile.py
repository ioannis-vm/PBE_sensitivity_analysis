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
                   for i in range(8)]

gms = ['gm'+str(i+1) for i in range(14)]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mkf = Makefile()

mkf.add_preamble("num_levels=3")

mkf.add_rule(
    'all',
    ['make/all_ground_motions_parsed',
     'make/all_ground_motion_figures_merged',
     'make/all_responses_obtained',
     'make/all_response_figures_obtained',
     'make/response_figures_merged',
     'make/all_response_TH_figures_obtained',
     'make/all_performance_evals_obtained',
     'make/all_performance_figures_obtained',
     'make/performance_figures_merged'],
    [])


mkf.add_rule(
    'clean',
    '',
    ['rm -rf analysis/*/ground_motions/parsed',
     'rm -rf figures/*',
     'rm -rf analysis/*/response',
     'rm -rf analysis/*/response_summary',
     'rm -rf analysis/*/performance',
     'rm -rf analysis/*/pelicun_log.txt',
     'rm -rf make/*'],
    phony=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Parsing raw PEER ground motion files #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

mkf.add_rule(
    "make/all_ground_motions_parsed",
    ["make/"+hz+"/ground_motions/ground_motions_parsed"
     for hz in hazard_lvl_dirs],
    ["touch make/all_ground_motions_parsed"])

for hz in hazard_lvl_dirs:
    input_dir = 'analysis/' + hz + '/ground_motions/peer_raw'
    output_dir = 'analysis/' + hz + '/ground_motions/parsed'
    plot_dir = 'figures/' + hz + '/ground_motions'
    mkf.add_rule(
        "make/"+hz+"/ground_motions/ground_motions_parsed",
        [
            'analysis/'+hz+'/ground_motions/peer_raw/_SearchResults.csv',
            'src/parse_gms.py',
            'src/ground_motion_utils.py'],
        [
            "python src/parse_gms.py '--input_dir' '"+input_dir+"' '--output_dir' '"+output_dir+"' '--plot_dir' '"+plot_dir+"'",
            "mkdir -p make/"+hz+"/ground_motions && touch make/"+hz+"/ground_motions/ground_motions_parsed"
        ])

# merging of generated ground motion figures
mkf.add_rule(
    "make/all_ground_motion_figures_merged",
    ["make/all_ground_motions_parsed"],
    ["bash -c \"mkdir -p figures/merged/ground_motions && gs -q -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE=figures/merged/ground_motions/RS.pdf -dBATCH $$(find . -name 'RS.pdf' | sort | tr '\\n' ' ' | sed -r 's/\.\///g')\" && bash -c \"mkdir -p figures/merged/ground_motions && gs -q -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE=figures/merged/ground_motions/time_history.pdf -dBATCH $$(find . -name 'time_history.pdf' | sort | tr '\\n' ' ' | sed -r 's/\.\///g')\" && touch make/all_ground_motion_figures_merged"])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Running NLTH analysis to get the building response #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

mkf.add_rule(
    "make/all_responses_obtained",
    ["make/"+hz+"/response/all_responses_obtained"
     for hz in hazard_lvl_dirs],
    [])

for hz in hazard_lvl_dirs:
    mkf.add_rule(
        "make/"+hz+"/response/all_responses_obtained",
        ["make/"+hz+"/response/"+gm+"/response_obtained"
         for hz in hazard_lvl_dirs
         for gm in gms] +
        ['src/response_vectors.py'],
        ["python src/response_vectors.py '--input_dir' 'analysis/"+hz+"/response' '--output_dir' 'analysis/"+hz+"/response_summary' '--num_levels' '$(num_levels)' '--num_inputs' '14' && touch make/"+hz+"/response/all_responses_obtained"])

for hz in hazard_lvl_dirs:
    for gm in gms:
        mkf.add_rule(
            "make/"+hz+"/response/"+gm+"/response_obtained",
            [
                'make/'+hz+'/ground_motions/ground_motions_parsed',
                'src/response.py',
            ],
            [
                "python src/response.py '--gm_dir' 'analysis/"+hz+"/ground_motions/parsed' '--gm_dt' '0.005' '--analysis_dt' '0.01' '--gm_number' '"+gm+"' '--output_dir' 'analysis/"+hz+"/response/"+gm+"' && mkdir -p make/"+hz+"/response/"+gm+" && touch make/"+hz+"/response/"+gm+"/response_obtained"
            ])
        # mkf.add_rule(
        #     "make/"+hz+"/response/"+gm+"/response_obtained",
        #     [
        #         'make/'+hz+'/ground_motions/ground_motions_parsed',
        #         'src/response.py',
        #     ],
        #     [
        #         "mkdir -p make/"+hz+"/response/"+gm+" && touch make/"+hz+"/response/"+gm+"/response_obtained"
        #     ])

# response figures

mkf.add_rule(
    "make/all_response_TH_figures_obtained",
    ["make/"+hz+"/response/response_TH_figures_obtained"
     for hz in hazard_lvl_dirs],
    ["touch make/all_response_TH_figures_obtained"])

mkf.write_to_file('Makefile')
for hz in hazard_lvl_dirs:
    mkf.add_rule(
        "make/"+hz+"/response/response_TH_figures_obtained",
        ["figures/"+hz+"/response/"+gm+"/FA.pdf" for gm in gms] +
        ["figures/"+hz+"/response/"+gm+"/ID.pdf" for gm in gms] +
        ["figures/"+hz+"/response/"+gm+"/FV.pdf" for gm in gms],
        ["touch make/"+hz+"/response/response_TH_figures_obtained"])

for hz in hazard_lvl_dirs:
    mkf.add_rule(
        "figures/"+hz+"/response/gm%/FA.pdf",
        ["make/"+hz+"/response/all_responses_obtained",
         "src/response_figures_TH.py"],
        ["mkdir -p figures/"+hz+"/response/gm$* && python src/response_figures_TH.py '--input_dir' 'analysis/"+hz+"/response/gm$*' '--fig_type' 'FA' '--output_filename' 'figures/"+hz+"/response/gm$*/FA.pdf' '--num_levels' '$(num_levels)'"])
    mkf.add_rule(
        "figures/"+hz+"/response/gm%/ID.pdf",
        ["make/"+hz+"/response/all_responses_obtained",
         "src/response_figures_TH.py"],
        ["mkdir -p figures/"+hz+"/response/gm$* && python src/response_figures_TH.py '--input_dir' 'analysis/"+hz+"/response/gm$*' '--fig_type' 'ID' '--output_filename' 'figures/"+hz+"/response/gm$*/ID.pdf' '--num_levels' '$(num_levels)'"])
    mkf.add_rule(
        "figures/"+hz+"/response/gm%/FV.pdf",
        ["make/"+hz+"/response/all_responses_obtained",
         "src/response_figures_TH.py"],
        ["mkdir -p figures/"+hz+"/response/gm$* && python src/response_figures_TH.py '--input_dir' 'analysis/"+hz+"/response/gm$*' '--fig_type' 'FV' '--output_filename' 'figures/"+hz+"/response/gm$*/FV.pdf' '--num_levels' '$(num_levels)'"])


mkf.add_rule(
    "make/all_response_figures_obtained",
    ["make/"+hz+"/response/response_figures_obtained"
     for hz in hazard_lvl_dirs],
    ["touch make/all_response_figures_obtained"])

for hz in hazard_lvl_dirs:
    mkf.add_rule(
        "make/"+hz+"/response/response_figures_obtained",
        ["figures/"+hz+"/response/PID-1.pdf",
         "figures/"+hz+"/response/PID-2.pdf",
         "figures/"+hz+"/response/PFA-1.pdf",
         "figures/"+hz+"/response/PFA-2.pdf",
         "figures/"+hz+"/response/PFA_norm-1.pdf",
         "figures/"+hz+"/response/PFA_norm-2.pdf",
         "figures/"+hz+"/response/PFV-1.pdf",
         "figures/"+hz+"/response/PFV-2.pdf"],
        ["touch make/"+hz+"/response/response_figures_obtained"])

for direction in ['1', '2']:
    mkf.add_rule(
        "figures/hazard_level_%/response/PID-"+direction+".pdf",
        ["src/response_figures.py",
         "make/hazard_level_%/response/all_responses_obtained"],
        ["python src/response_figures.py '--figure_type' 'PID' '--direction' '"+direction+"' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'"])
    mkf.add_rule(
        "figures/hazard_level_%/response/PFA-"+direction+".pdf",
        ["src/response_figures.py",
         "make/hazard_level_%/response/all_responses_obtained"],
        ["python src/response_figures.py '--figure_type' 'PFA' '--direction' '"+direction+"' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'"])
    mkf.add_rule(
        "figures/hazard_level_%/response/PFA_norm-"+direction+".pdf",
        ["src/response_figures.py",
         "make/hazard_level_%/response/all_responses_obtained"],
        ["python src/response_figures.py '--figure_type' 'PFA_norm' '--direction' '"+direction+"' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'"])
    mkf.add_rule(
        "figures/hazard_level_%/response/PFV-"+direction+".pdf",
        ["src/response_figures.py",
         "make/hazard_level_%/response/all_responses_obtained"],
        ["python src/response_figures.py '--figure_type' 'PFV' '--direction' '"+direction+"' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'"])

# merging response figures to a single pdf file

mkf.add_rule(
    "make/response_figures_merged",
    ["figures/merged/response/PFA-1.pdf",
     "figures/merged/response/PFA-2.pdf",
     "figures/merged/response/PFA_norm-1.pdf",
     "figures/merged/response/PFA_norm-2.pdf",
     "figures/merged/response/PFV-1.pdf",
     "figures/merged/response/PFV-2.pdf",
     "figures/merged/response/PID-1.pdf",
     "figures/merged/response/PID-2.pdf"],
    ["touch make/response_figures_merged"])

mkf.add_rule(
    "figures/merged/response/%",
    ["make/all_response_figures_obtained"],
    ["bash -c \"mkdir -p figures/merged/response && gs -q -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE=figures/merged/response/$* -dBATCH $$(find . -name '$*' | sort | tr '\\n' ' ' | sed -r 's/\.\///g')\""])



# ~~~~~~~~~~~~~~~~~~~~~~ #
# Performance Evaluation #
# ~~~~~~~~~~~~~~~~~~~~~~ #

mkf.add_rule(
    "make/all_performance_evals_obtained",
    [f"make/{hz}/performance/performance_evals_obtained" for hz in hazard_lvl_dirs],
    ["touch make/all_performance_evals_obtained"])

for hz in hazard_lvl_dirs:
    prereqs = ["make/"+hz+"/response/all_responses_obtained",
               "src/performance_eval.py",
               "src/performance_data/CMP_marginals.csv",
               "src/performance_data/fragility_Additional.csv",
               "src/performance_data/LOSS_map.csv",
               "src/performance_data/repair_Additional.csv",
               "src/performance_data/resources_modified/fragility_DB_FEMA_P58_2nd.csv",
               "src/performance_data/resources_modified/bldg_repair_DB_FEMA_P58_2nd.csv"
               ]
    mkf.add_rule(
        "make/"+hz+"/performance/performance_evals_obtained",
        ["make/"+hz+"/performance/"+x+"/performance_eval_obtained" for x in ['0', 'A', 'B', 'C', 'D', 'E', 'F']],
        ["mkdir -p make/"+hz+"/performance && touch make/"+hz+"/performance/performance_evals_obtained"])
    mkf.add_rule(
        "make/"+hz+"/performance/0/performance_eval_obtained",
        prereqs,
        ["mkdir -p analysis/"+hz+"/performance/0 && python src/performance_eval.py '--response_path' 'analysis/"+hz+"/response_summary/response.csv' '--output_directory' 'analysis/"+hz+"/performance/0' '--c_edp_stdev' '1.00' '--c_quant_stdev' '1.00' '--c_dm_stdev' '1.00' '--c_collapse_stdev' '1.00' '--c_dv_stdev' '1.00' '--c_replace_stdev' '1.00' && mkdir -p make/"+hz+"/performance/0 && touch make/"+hz+"/performance/0/performance_eval_obtained"])
    mkf.add_rule(
        "make/"+hz+"/performance/A/performance_eval_obtained",
        prereqs,
        ["mkdir -p analysis/"+hz+"/performance/A && python src/performance_eval.py '--response_path' 'analysis/"+hz+"/response_summary/response.csv' '--output_directory' 'analysis/"+hz+"/performance/A' '--c_edp_stdev' '0.001' '--c_quant_stdev' '1.00' '--c_dm_stdev' '1.00' '--c_collapse_stdev' '1.00' '--c_dv_stdev' '1.00' '--c_replace_stdev' '1.00' && mkdir -p make/"+hz+"/performance/A  && touch make/"+hz+"/performance/A/performance_eval_obtained"])
    mkf.add_rule(
        "make/"+hz+"/performance/B/performance_eval_obtained",
        prereqs,
        ["mkdir -p analysis/"+hz+"/performance/B && python src/performance_eval.py '--response_path' 'analysis/"+hz+"/response_summary/response.csv' '--output_directory' 'analysis/"+hz+"/performance/B' '--c_edp_stdev' '1.00' '--c_quant_stdev' '0.001' '--c_dm_stdev' '1.00' '--c_collapse_stdev' '1.00' '--c_dv_stdev' '1.00' '--c_replace_stdev' '1.00' && mkdir -p make/"+hz+"/performance/B  && touch make/"+hz+"/performance/B/performance_eval_obtained"])
    mkf.add_rule(
        "make/"+hz+"/performance/C/performance_eval_obtained",
        prereqs,
        ["mkdir -p analysis/"+hz+"/performance/C && python src/performance_eval.py '--response_path' 'analysis/"+hz+"/response_summary/response.csv' '--output_directory' 'analysis/"+hz+"/performance/C' '--c_edp_stdev' '1.00' '--c_quant_stdev' '1.00' '--c_dm_stdev' '0.001' '--c_collapse_stdev' '1.00' '--c_dv_stdev' '1.00' '--c_replace_stdev' '1.00' && mkdir -p make/"+hz+"/performance/C  && touch make/"+hz+"/performance/C/performance_eval_obtained"])
    mkf.add_rule(
        "make/"+hz+"/performance/D/performance_eval_obtained",
        prereqs,
        ["mkdir -p analysis/"+hz+"/performance/D && python src/performance_eval.py '--response_path' 'analysis/"+hz+"/response_summary/response.csv' '--output_directory' 'analysis/"+hz+"/performance/D' '--c_edp_stdev' '1.00' '--c_quant_stdev' '1.00' '--c_dm_stdev' '1.00' '--c_collapse_stdev' '0.001' '--c_dv_stdev' '1.00' '--c_replace_stdev' '1.00' && mkdir -p make/"+hz+"/performance/D  && touch make/"+hz+"/performance/D/performance_eval_obtained"])
    mkf.add_rule(
        "make/"+hz+"/performance/E/performance_eval_obtained",
        prereqs,
        ["mkdir -p analysis/"+hz+"/performance/E && python src/performance_eval.py '--response_path' 'analysis/"+hz+"/response_summary/response.csv' '--output_directory' 'analysis/"+hz+"/performance/E' '--c_edp_stdev' '1.00' '--c_quant_stdev' '1.00' '--c_dm_stdev' '1.00' '--c_collapse_stdev' '1.00' '--c_dv_stdev' '0.001' '--c_replace_stdev' '1.00' && mkdir -p make/"+hz+"/performance/E  && touch make/"+hz+"/performance/E/performance_eval_obtained"])
    mkf.add_rule(
        "make/"+hz+"/performance/F/performance_eval_obtained",
        prereqs,
        ["mkdir -p analysis/"+hz+"/performance/F && python src/performance_eval.py '--response_path' 'analysis/"+hz+"/response_summary/response.csv' '--output_directory' 'analysis/"+hz+"/performance/F' '--c_edp_stdev' '1.00' '--c_quant_stdev' '1.00' '--c_dm_stdev' '1.00' '--c_collapse_stdev' '1.00' '--c_dv_stdev' '1.00' '--c_replace_stdev' '0.001' && mkdir -p make/"+hz+"/performance/F  && touch make/"+hz+"/performance/F/performance_eval_obtained"])

# performance evaluation figures

mkf.add_rule(
    "make/all_performance_figures_obtained",
    ["make/"+hz+"/performance_figures_obtained"
     for hz in hazard_lvl_dirs],
    ["touch make/all_performance_figures_obtained"])

for hz in hazard_lvl_dirs:
    # mkf.add_rule(
    #     "make/"+hz+"/performance_figures_obtained",
    #     ["figures/"+hz+"/performance/total_cost_"+x+".pdf"
    #      for x in ["A-B", "A-C", "A-D", "A-E", "A-F"]] +
    #     ["figures/"+hz+"/performance/component_costs_"+x+".html"
    #      for x in ["A", "B", "C", "D", "E", "F"]] +
    #     ["figures/"+hz+"/performance/damage_states_"+x+".pdf"
    #      for x in ["A", "B", "C", "D", "E", "F"]] +
    #     ["figures/"+hz+"/performance/cost_deagg_"+x+".pdf"
    #      for x in ["A", "B", "C", "D", "E", "F"]],
    #     ["touch make/"+hz+"/performance_figures_obtained"])
    mkf.add_rule(
        "make/"+hz+"/performance_figures_obtained",
        [f"figures/{hz}/performance/0/repair_per_category.pdf"],
        ["touch make/"+hz+"/performance_figures_obtained"])
    mkf.add_rule(
        f"figures/{hz}/performance/0/repair_per_category.pdf",
        ["src/performance_figures/cost_deagg_group.py",
         f"make/{hz}/performance/performance_evals_obtained"],
        [f"python src/performance_figures/cost_deagg_group.py '--LOSS_repair_path' 'analysis/{hz}/performance/0/LOSS_repair.csv' '--output_path' 'figures/{hz}/performance/0'"])

# merging performance figures to a single pdf file

# mkf.add_rule(
#     "make/performance_figures_merged",
#     ["figures/merged/performance/total_cost_A-B.pdf",
#      "figures/merged/performance/total_cost_A-C.pdf",
#      "figures/merged/performance/total_cost_A-D.pdf",
#      "figures/merged/performance/total_cost_A-E.pdf",
#      "figures/merged/performance/total_cost_A-F.pdf",
#      "figures/merged/performance/damage_states_A.pdf",
#      "figures/merged/performance/damage_states_C.pdf",
#      "figures/merged/performance/cost_deagg_A.pdf"],
#     ["touch make/performance_figures_merged"])








mkf.write_to_file('Makefile')
