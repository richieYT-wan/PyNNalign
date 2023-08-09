#! /usr/bin/bash

epochs=100
motif_length=9
hidden_neurons=50

/home/projects/vaccine/people/morni/nnalign-2.1/bin/Linux_x86_64/nnalign_gaps_pan-2.1a -bls 50 -a ARNDCQEGHILKMFPSTWYV -eplen 0 -elpfr 0 -i ${epochs} -l ${motif_length}  -s 2 -burn 10 -gl 0 -il 0 -nh ${hidden_neurons} -fl 3 -blf /home/projects/vaccine/people/morni/nnalign-2.1/data/blosum62.freq_rownorm  -mpat /home/projects/vaccine/people/morni/nnalign-2.1/data/BLOSUM%i -syn /home/projects/vaccine/people/morni/nnalign-2.1/test/HLA_example_17213.tmp/syn/0.10.bl.2.lg9.syn -ft /home/projects/vaccine/people/yatwan/PyNNalign/data/NetMHCIIpan_train/benchmark_test.txt -bl -eta 0.05 /home/projects/vaccine/people/yatwan/PyNNalign/data/NetMHCIIpan_train/benchmark_train.txt

