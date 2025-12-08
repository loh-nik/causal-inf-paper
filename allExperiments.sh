#!/bin/bash
python Pipeline.py -worldMap True -methods "PCMCI" "LKIF" -indep_test "parcorr" -map_var_names "AMOC" "ASSI" "Temp" -filename_in "./DataSheet SeaIceAggregates.csv" -dirname_out "results_paper_confounderDetrend_marchSeptMask" -vars "AMOC_Caesar" "SI_Conc_MostChange_DetrDes" "Arctic_Temp_Anomalies" -vars_names "AMOC" "ASSI" "Temp" -mask "MarchSept" "MarchSept" "MarchSept" -maskType "x" -mask_lkif "MarchSept" -tauMax 5 -alpha 0.05 -detrend False False True -deseason False False True

for var in \
    AMOC_hadcrut5 \
    AMOC_hadsst4 \
    AMOC_cobe \
    AMOC_ersst \
    AMOC_dip_era5 \
    AMOC_dip_hadcrut5 \
    AMOC_dip_hadsst4 \
    AMOC_dip_cobe \
    AMOC_dip_ersst
do
    python Pipeline.py -worldMap True -methods "PCMCI" "LKIF" -indep_test "parcorr" -map_var_names "$var" "ASSI" "Temp" -filename_in "./DataSheet SeaIceAggregates.csv" -dirname_out "results_paper_${var}" -vars "$var" "SI_Conc_MostChange_DetrDes" "Arctic_Temp_Anomalies" -vars_names "$var" "ASSI" "Temp" -mask "MarchSept" "MarchSept" "MarchSept" -maskType "x" -mask_lkif "MarchSept" -tauMax 5 -alpha 0.05 -detrend False False True -deseason False False True
done