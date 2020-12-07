#!/bin/sh
python3 scripts/eval_bop19.py --renderer_type=python --result_filenames=mark-sensors-mask_lm-data.csv
#python3 scripts/vis_est_poses.py --result_filenames=mark-sensors-mask_lm-data.csv
#python3 scripts/show_performance_bop19.py --result_filenames=mark-sensors-mask_lm-data.csv
