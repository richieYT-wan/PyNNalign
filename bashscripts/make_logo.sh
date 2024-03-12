#! /usr/bin/bash

# This creates logos for all files located within a directory, saving it with the same basename as the input
# Replace the conda.sh with your own path
# the first argument ${1} is the input folder ; ${2} is the output folder
# example of usage : sh make_logo.sh ./input_folder/ ../path/to/output_folder/

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate py27

S2L="/home/projects/vaccine/people/morni/seq2logo-2.1/seq2logo"
DATADIR="$(realpath ${1})/"
ODIR="$(realpath ${2})/"

task(){
        BASENAME=$(basename ${ODIR}${1} | sed 's/\.[^.]*$//')
        mkdir -p ${ODIR}${BASENAME}
        echo "here" ${BASENAME}
        $(${S2L} -f ${DATADIR}${1} -o "${ODIR}${BASENAME}/${BASENAME}" -t ${BASENAME})
}
for f in $(ls ${DATADIR})
do
        task ${f}
done