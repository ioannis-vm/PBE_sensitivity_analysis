num_levels=3

all :\
ground_motions\
distribute_makefiles\
src/building.pcl\
distr_response\
distr_response_summary\
response_figures\
distr_performance\
distr_performance_figures\
performance_figures_combined

clean:\
clean_ground_motions\
remove_makefiles\
distr_clean_response\
distr_clean_response_summary\
clean_response_figures\
distr_clean_performance\
distr_clean_performance_figures

.PHONY:\
all\
ground_motions\
clean_ground_motions\
distribute_makefiles\
remove_makefiles\
clean_response\
response_summary\
clean_response_summary\
response_figures\
clean_response_figures\
performance\
distr_clean_performance\
performance_figures\
distr_clean_performance_figures

ground_motions :\
analysis/hazard_level_1/ground_motions/parsed/empty_target_file\
analysis/hazard_level_2/ground_motions/parsed/empty_target_file\
analysis/hazard_level_3/ground_motions/parsed/empty_target_file\
analysis/hazard_level_4/ground_motions/parsed/empty_target_file\
analysis/hazard_level_5/ground_motions/parsed/empty_target_file\
analysis/hazard_level_6/ground_motions/parsed/empty_target_file\
analysis/hazard_level_7/ground_motions/parsed/empty_target_file\
analysis/hazard_level_8/ground_motions/parsed/empty_target_file\
analysis/hazard_level_test/ground_motions/parsed/empty_target_file

analysis/%/ground_motions/parsed/empty_target_file : analysis/%/ground_motions/peer_raw/_SearchResults.csv src/parse_gms.py src/ground_motion_utils.py
	python src/parse_gms.py '--input_dir' 'analysis/$*/ground_motions/peer_raw' '--output_dir' 'analysis/$*/ground_motions/parsed' '--plot_dir' 'figures/$*/ground_motions'
	touch analysis/$*/ground_motions/parsed/empty_target_file

clean_ground_motions:
	@cd analysis/hazard_level_1/ground_motions && rm -rf parsed
	@cd analysis/hazard_level_2/ground_motions && rm -rf parsed
	@cd analysis/hazard_level_3/ground_motions && rm -rf parsed
	@cd analysis/hazard_level_4/ground_motions && rm -rf parsed
	@cd analysis/hazard_level_5/ground_motions && rm -rf parsed
	@cd analysis/hazard_level_6/ground_motions && rm -rf parsed
	@cd analysis/hazard_level_7/ground_motions && rm -rf parsed
	@cd analysis/hazard_level_8/ground_motions && rm -rf parsed

distribute_makefiles: analysis/makefiles_distributed

analysis/makefiles_distributed:\
analysis/hazard_level_1/Makefile\
analysis/hazard_level_2/Makefile\
analysis/hazard_level_3/Makefile\
analysis/hazard_level_4/Makefile\
analysis/hazard_level_5/Makefile\
analysis/hazard_level_6/Makefile\
analysis/hazard_level_7/Makefile\
analysis/hazard_level_8/Makefile\
analysis/hazard_level_test/Makefile
	@touch analysis/makefiles_distributed

analysis/hazard_level_%/Makefile: src/Makefile_hz
	@cp src/Makefile_hz analysis/hazard_level_$*/Makefile

remove_makefiles:
	@rm -rf analysis/makefiles_distributed
	@rm -rf analysis/hazard_level_1/Makefile
	@rm -rf analysis/hazard_level_2/Makefile
	@rm -rf analysis/hazard_level_3/Makefile
	@rm -rf analysis/hazard_level_4/Makefile
	@rm -rf analysis/hazard_level_5/Makefile
	@rm -rf analysis/hazard_level_6/Makefile
	@rm -rf analysis/hazard_level_7/Makefile
	@rm -rf analysis/hazard_level_8/Makefile
	@rm -rf analysis/hazard_level_test/Makefile


src/building.pcl : src/design.py
	python src/design.py '--output_path' 'src/building.pcl'



distr_%: analysis/makefiles_distributed
	@cd analysis/hazard_level_1 && $(MAKE) distr_$*
	@cd analysis/hazard_level_2 && $(MAKE) distr_$*
	@cd analysis/hazard_level_3 && $(MAKE) distr_$*
	@cd analysis/hazard_level_4 && $(MAKE) distr_$*
	@cd analysis/hazard_level_5 && $(MAKE) distr_$*
	@cd analysis/hazard_level_6 && $(MAKE) distr_$*
	@cd analysis/hazard_level_7 && $(MAKE) distr_$*
	@cd analysis/hazard_level_8 && $(MAKE) distr_$*
	@cd analysis/hazard_level_test && $(MAKE) distr_$*


response_figures:\
figures/hazard_level_1/response/PID-1.pdf\
figures/hazard_level_2/response/PID-1.pdf\
figures/hazard_level_3/response/PID-1.pdf\
figures/hazard_level_4/response/PID-1.pdf\
figures/hazard_level_5/response/PID-1.pdf\
figures/hazard_level_6/response/PID-1.pdf\
figures/hazard_level_7/response/PID-1.pdf\
figures/hazard_level_8/response/PID-1.pdf\
figures/hazard_level_test/response/PID-1.pdf\
figures/hazard_level_1/response/PID-2.pdf\
figures/hazard_level_2/response/PID-2.pdf\
figures/hazard_level_3/response/PID-2.pdf\
figures/hazard_level_4/response/PID-2.pdf\
figures/hazard_level_5/response/PID-2.pdf\
figures/hazard_level_6/response/PID-2.pdf\
figures/hazard_level_7/response/PID-2.pdf\
figures/hazard_level_8/response/PID-2.pdf\
figures/hazard_level_test/response/PID-2.pdf\
figures/hazard_level_1/response/PFA-1.pdf\
figures/hazard_level_2/response/PFA-1.pdf\
figures/hazard_level_3/response/PFA-1.pdf\
figures/hazard_level_4/response/PFA-1.pdf\
figures/hazard_level_5/response/PFA-1.pdf\
figures/hazard_level_6/response/PFA-1.pdf\
figures/hazard_level_7/response/PFA-1.pdf\
figures/hazard_level_8/response/PFA-1.pdf\
figures/hazard_level_test/response/PFA-1.pdf\
figures/hazard_level_1/response/PFA-2.pdf\
figures/hazard_level_2/response/PFA-2.pdf\
figures/hazard_level_3/response/PFA-2.pdf\
figures/hazard_level_4/response/PFA-2.pdf\
figures/hazard_level_5/response/PFA-2.pdf\
figures/hazard_level_6/response/PFA-2.pdf\
figures/hazard_level_7/response/PFA-2.pdf\
figures/hazard_level_8/response/PFA-2.pdf\
figures/hazard_level_test/response/PFA-2.pdf\
figures/hazard_level_1/response/PFV-1.pdf\
figures/hazard_level_2/response/PFV-1.pdf\
figures/hazard_level_3/response/PFV-1.pdf\
figures/hazard_level_4/response/PFV-1.pdf\
figures/hazard_level_5/response/PFV-1.pdf\
figures/hazard_level_6/response/PFV-1.pdf\
figures/hazard_level_7/response/PFV-1.pdf\
figures/hazard_level_8/response/PFV-1.pdf\
figures/hazard_level_test/response/PFV-1.pdf\
figures/hazard_level_1/response/PFV-2.pdf\
figures/hazard_level_2/response/PFV-2.pdf\
figures/hazard_level_3/response/PFV-2.pdf\
figures/hazard_level_4/response/PFV-2.pdf\
figures/hazard_level_5/response/PFV-2.pdf\
figures/hazard_level_6/response/PFV-2.pdf\
figures/hazard_level_7/response/PFV-2.pdf\
figures/hazard_level_8/response/PFV-2.pdf\
figures/hazard_level_test/response/PFV-2.pdf


figures/hazard_level_%/response/PID-1.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PID' '--direction' '1' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

figures/hazard_level_%/response/PID-2.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PID' '--direction' '2' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

figures/hazard_level_%/response/PFA-1.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PFA' '--direction' '1' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

figures/hazard_level_%/response/PFA-2.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PFA' '--direction' '2' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

figures/hazard_level_%/response/PFV-1.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PFV' '--direction' '1' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

figures/hazard_level_%/response/PFV-2.pdf: src/response_figures.py analysis/hazard_level_%/response/all_responses_obtained
	@python src/response_figures.py '--figure_type' 'PFV' '--direction' '2' '--input_dir' 'analysis/hazard_level_$*/response' '--output_dir' 'figures/hazard_level_$*/response' '--num_levels' '$(num_levels)'

clean_response_figures:
	@rm -rf figures/hazard_level_1/response
	@rm -rf figures/hazard_level_2/response
	@rm -rf figures/hazard_level_3/response
	@rm -rf figures/hazard_level_4/response
	@rm -rf figures/hazard_level_5/response
	@rm -rf figures/hazard_level_6/response
	@rm -rf figures/hazard_level_7/response
	@rm -rf figures/hazard_level_8/response
	@rm -rf figures/hazard_level_test/response

performance_figures_combined:\
analysis/hazard_level_1/performance/A/DL_summary.csv\
analysis/hazard_level_1/performance/C/DL_summary.csv\
analysis/hazard_level_2/performance/A/DL_summary.csv\
analysis/hazard_level_2/performance/C/DL_summary.csv\
analysis/hazard_level_3/performance/A/DL_summary.csv\
analysis/hazard_level_3/performance/C/DL_summary.csv\
analysis/hazard_level_4/performance/A/DL_summary.csv\
analysis/hazard_level_4/performance/C/DL_summary.csv\
analysis/hazard_level_5/performance/A/DL_summary.csv\
analysis/hazard_level_5/performance/C/DL_summary.csv\
analysis/hazard_level_6/performance/A/DL_summary.csv\
analysis/hazard_level_6/performance/C/DL_summary.csv\
analysis/hazard_level_7/performance/A/DL_summary.csv\
analysis/hazard_level_7/performance/C/DL_summary.csv\
analysis/hazard_level_8/performance/A/DL_summary.csv\
analysis/hazard_level_8/performance/C/DL_summary.csv
	@mkdir -p figures/combined/performance && python src/performance_figures/combined.py
