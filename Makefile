num_levels=3

all :\
distribute_makefiles\
distr_ground_motions\
distr_response\
distr_response_summary\
response_figures\
distr_response_TH_figures\
distr_response_TH_figures_merged\
distr_performance\
distr_performance_figures\
response_figures_merged\
performance_figures_merged

clean: clean_makefiles

clean_data_complete:
	@rm -rf analysis/*/ground_motions/parsed
	@rm -rf figures/*
	@rm -rf analysis/*/response
	@rm -rf analysis/*/response_summary
	@rm -rf analysis/*/performance
	@rm -rf analysis/*/pelicun_log.txt
	@touch clean_data_complete

clean_makefiles: clean_data_complete
	@rm -rf clean_data_complete
	@rm -rf analysis/makefiles_distributed
	@rm -rf analysis/hazard_level_*/Makefile

.PHONY:\
all\
clean\
distribute_makefiles\
clean_makefiles\
performance\
performance_figures\

distribute_makefiles: analysis/makefiles_distributed

analysis/makefiles_distributed:\
analysis/hazard_level_1/Makefile\
analysis/hazard_level_2/Makefile\
analysis/hazard_level_3/Makefile\
analysis/hazard_level_4/Makefile\
analysis/hazard_level_5/Makefile\
analysis/hazard_level_6/Makefile\
analysis/hazard_level_7/Makefile\
analysis/hazard_level_8/Makefile
	@touch analysis/makefiles_distributed

analysis/hazard_level_%/Makefile: src/Makefile_hz
	@cp src/Makefile_hz analysis/hazard_level_$*/Makefile

distr_%: analysis/makefiles_distributed
	@cd analysis/hazard_level_1 && $(MAKE) distr_$*
	@cd analysis/hazard_level_2 && $(MAKE) distr_$*
	@cd analysis/hazard_level_3 && $(MAKE) distr_$*
	@cd analysis/hazard_level_4 && $(MAKE) distr_$*
	@cd analysis/hazard_level_5 && $(MAKE) distr_$*
	@cd analysis/hazard_level_6 && $(MAKE) distr_$*
	@cd analysis/hazard_level_7 && $(MAKE) distr_$*
	@cd analysis/hazard_level_8 && $(MAKE) distr_$*
	@touch distr_$*

response_figures: response_figures_generated

response_figures_generated:\
figures/hazard_level_1/response/PID-1.pdf\
figures/hazard_level_2/response/PID-1.pdf\
figures/hazard_level_3/response/PID-1.pdf\
figures/hazard_level_4/response/PID-1.pdf\
figures/hazard_level_5/response/PID-1.pdf\
figures/hazard_level_6/response/PID-1.pdf\
figures/hazard_level_7/response/PID-1.pdf\
figures/hazard_level_8/response/PID-1.pdf\
figures/hazard_level_1/response/PID-2.pdf\
figures/hazard_level_2/response/PID-2.pdf\
figures/hazard_level_3/response/PID-2.pdf\
figures/hazard_level_4/response/PID-2.pdf\
figures/hazard_level_5/response/PID-2.pdf\
figures/hazard_level_6/response/PID-2.pdf\
figures/hazard_level_7/response/PID-2.pdf\
figures/hazard_level_8/response/PID-2.pdf\
figures/hazard_level_1/response/PFA-1.pdf\
figures/hazard_level_2/response/PFA-1.pdf\
figures/hazard_level_3/response/PFA-1.pdf\
figures/hazard_level_4/response/PFA-1.pdf\
figures/hazard_level_5/response/PFA-1.pdf\
figures/hazard_level_6/response/PFA-1.pdf\
figures/hazard_level_7/response/PFA-1.pdf\
figures/hazard_level_8/response/PFA-1.pdf\
figures/hazard_level_1/response/PFA-2.pdf\
figures/hazard_level_2/response/PFA-2.pdf\
figures/hazard_level_3/response/PFA-2.pdf\
figures/hazard_level_4/response/PFA-2.pdf\
figures/hazard_level_5/response/PFA-2.pdf\
figures/hazard_level_6/response/PFA-2.pdf\
figures/hazard_level_7/response/PFA-2.pdf\
figures/hazard_level_8/response/PFA-2.pdf\
figures/hazard_level_1/response/PFA_norm-1.pdf\
figures/hazard_level_2/response/PFA_norm-1.pdf\
figures/hazard_level_3/response/PFA_norm-1.pdf\
figures/hazard_level_4/response/PFA_norm-1.pdf\
figures/hazard_level_5/response/PFA_norm-1.pdf\
figures/hazard_level_6/response/PFA_norm-1.pdf\
figures/hazard_level_7/response/PFA_norm-1.pdf\
figures/hazard_level_8/response/PFA_norm-1.pdf\
figures/hazard_level_1/response/PFA_norm-2.pdf\
figures/hazard_level_2/response/PFA_norm-2.pdf\
figures/hazard_level_3/response/PFA_norm-2.pdf\
figures/hazard_level_4/response/PFA_norm-2.pdf\
figures/hazard_level_5/response/PFA_norm-2.pdf\
figures/hazard_level_6/response/PFA_norm-2.pdf\
figures/hazard_level_7/response/PFA_norm-2.pdf\
figures/hazard_level_8/response/PFA_norm-2.pdf\
figures/hazard_level_1/response/PFV-1.pdf\
figures/hazard_level_2/response/PFV-1.pdf\
figures/hazard_level_3/response/PFV-1.pdf\
figures/hazard_level_4/response/PFV-1.pdf\
figures/hazard_level_5/response/PFV-1.pdf\
figures/hazard_level_6/response/PFV-1.pdf\
figures/hazard_level_7/response/PFV-1.pdf\
figures/hazard_level_8/response/PFV-1.pdf\
figures/hazard_level_1/response/PFV-2.pdf\
figures/hazard_level_2/response/PFV-2.pdf\
figures/hazard_level_3/response/PFV-2.pdf\
figures/hazard_level_4/response/PFV-2.pdf\
figures/hazard_level_5/response/PFV-2.pdf\
figures/hazard_level_6/response/PFV-2.pdf\
figures/hazard_level_7/response/PFV-2.pdf\
figures/hazard_level_8/response/PFV-2.pdf
	@touch response_figures_generated

figures/hazard_level_%/response/PID-1.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PID' '--direction' '1' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

figures/hazard_level_%/response/PID-2.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PID' '--direction' '2' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

figures/hazard_level_%/response/PFA-1.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PFA' '--direction' '1' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

figures/hazard_level_%/response/PFA-2.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PFA' '--direction' '2' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

figures/hazard_level_%/response/PFA_norm-1.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PFA_norm' '--direction' '1' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

figures/hazard_level_%/response/PFA_norm-2.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PFA_norm' '--direction' '2' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

figures/hazard_level_%/response/PFV-1.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PFV' '--direction' '1' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

figures/hazard_level_%/response/PFV-2.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PFV' '--direction' '2' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

response_figures_merged:\
figures/merged/response/PFA-1.pdf\
figures/merged/response/PFA-2.pdf\
figures/merged/response/PFA_norm-1.pdf\
figures/merged/response/PFA_norm-2.pdf\
figures/merged/response/PFV-1.pdf\
figures/merged/response/PFV-2.pdf\
figures/merged/response/PID-1.pdf\
figures/merged/response/PID-2.pdf


figures/merged/response/%: response_figures_generated
	@bash -c "mkdir -p figures/merged/response && gs -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE=figures/merged/response/$* -dBATCH $$(find . -name '$*' | sort | tr '\n' ' ' | sed -r 's/\.\///g')"








performance_figures_merged:\
figures/merged/performance/total_cost_A-B.pdf\
figures/merged/performance/total_cost_A-C.pdf\
figures/merged/performance/total_cost_A-D.pdf\
figures/merged/performance/total_cost_A-E.pdf\
figures/merged/performance/total_cost_A-F.pdf

figures/merged/performance/%: distr_performance_figures
	@bash -c "mkdir -p figures/merged/performance && gs -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE=figures/merged/performance/$* -dBATCH $$(find figures -name '$*' | sort | tr '\n' ' ')"
