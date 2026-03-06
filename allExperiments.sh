#!/bin/bash
python Pipeline.py -worldMap True -methods "PCMCI" "LKIF" -indep_test "parcorr" -map_var_names "AMOC" "ASSI" "Temp" -filename_in "./DataSheet SeaIceAggregates.csv" -dirname_out "results_paper_main" -vars "AMOC_Caesar" "SI_Conc" "Arctic_Temp_Anomalies" -vars_names "AMOC" "ASSI" "Temp" -mask "MarchSept" "MarchSept" "MarchSept" -maskType "x" -mask_lkif "MarchSept" -tauMax 5 -alpha 0.05 -detrend False False True -deseason False False True

# python Pipeline.py -methods "PCMCI" "LKIF" -indep_test "parcorr" -filename_in "./DataSheet SeaIceAggregates.csv" -dirname_out "results_paper_noConfound" -vars "AMOC_Caesar" "SI_Conc" -vars_names "AMOC" "ASSI" -mask "MarchSept" "MarchSept" -maskType "x" -mask_lkif "MarchSept" -tauMax 5 -alpha 0.05 -detrend False False -deseason False False

# for var in \
#     AMOC_hadcrut5 \
#     AMOC_hadsst4 \
#     AMOC_cobe \
#     AMOC_ersst
# do
#     python Pipeline.py -worldMap True -methods "PCMCI" "LKIF" -indep_test "parcorr" -map_var_names "$var" "ASSI" "Temp" -filename_in "./DataSheet SeaIceAggregates.csv" -dirname_out "results_paper_${var}" -vars "$var" "SI_Conc" "Arctic_Temp_Anomalies" -vars_names "$var" "ASSI" "Temp" -mask "MarchSept" "MarchSept" "MarchSept" -maskType "x" -mask_lkif "MarchSept" -tauMax 5 -alpha 0.05 -detrend False False True -deseason False False True
# done

# pairs=(
# "SI_Conc_0 ASSI_def_0.txt"
# "SI_Conc_50 ASSI_def_50.txt"
# "SI_Conc_75 ASSI_def_75.txt"
# "SI_Conc_90 ASSI_def_90.txt"
# )

# for pair in "${pairs[@]}"; do
#     read -r var var2 <<< "$pair"
#     python Pipeline.py -worldMap True -assi_file "$var2" -methods "PCMCI" "LKIF" -indep_test "parcorr" -map_var_names "AMOC" "$var" "Temp" -filename_in "./DataSheet SeaIceAggregates.csv" -dirname_out "results_paper_${var}" -vars "AMOC_Caesar" "$var" "Arctic_Temp_Anomalies" -vars_names "AMOC" "$var" "Temp" -mask "MarchSept" "MarchSept" "MarchSept" -maskType "x" -mask_lkif "MarchSept" -tauMax 5 -alpha 0.05 -detrend False False True -deseason False False True
# done