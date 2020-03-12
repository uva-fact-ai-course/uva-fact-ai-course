#!/bin/bash

export PYTHONPATH="$PWD"


ACC_path_tanh=$(ls -t Transparency/outputs/${1}/lstm+tanh)
ACC_path_dot=$(ls -t Transparency/outputs/${1}/lstm+dot)

stringarray1=($ACC_path_tanh)
none=${stringarray1[0]}
tanh=${stringarray1[1]}

stringarray2=($ACC_path_dot)
dot=${stringarray2[0]}


cd Transparency/outputs/${1}/lstm+tanh/${none}
none=$PWD

cd ../${tanh}
tanh=$PWD

cd ../../lstm+dot/${dot}
dot=$PWD


echo $none
echo $tanh
echo $dot
