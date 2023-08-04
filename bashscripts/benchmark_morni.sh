#! /usr/bin/bash

nnalign="/home/projects/vaccine/people/morni/nnalign-2.1/nnalign"
seq2logo="/home/projects/vaccine/people/morni/seq2logo-2.1/seq2logo"

motif_length=9
hidden_neurons=50
epochs=200
trainset=${1}
testset=${2}
name=${3}

 
${nnalign} -f /home/projects/vaccine/people/yatwan/PyNNalign/data/NetMHCIIpan_train/drb_concat.csv -name "${name}" -Logo ${seq2logo} -lgt ${motif_length} -nh ${hidden_neurons} -iter ${epochs} -encoding 1 -burn 10 -seeds 1 -bs 1 -rdir /home/projects/vaccine/people/yatwan/PyNNalign/output/230804_speed_benchmark_morni/ -split 3 -procs 1

	
