#!/bin/bash

O3=${1}
cnv=${2}
upw=${3}
CO2=${4}
RH=${5}
cntrl=${6}
cntrl2=${7}
user_cntrl=${8:-}

log_file=conv-O3_last-${RH}-${CO2}_${O3}${upw}${cnv}-${cntrl}-${cntrl2}

if [ -z "${user_cntrl}" ];
    then
        log_file=${log_file}
    else
        log_file=${log_file}_a
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${log_file}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --account=mh0066

source ~/.bash_profile_other3
cd ~/konrad_exps/running_scripts

python conv-O3_last.py ${O3} ${cnv} ${upw} ${CO2} ${RH} ${cntrl} ${cntrl2} ${user_cntrl} 2>&1 | tee ${log_file}.log

EOF

