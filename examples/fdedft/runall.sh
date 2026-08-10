#!/bin/bash

set -e

for example in 01_f_ch3_cl_uks
do
    echo "RUN: ${example}"
    (
        cd "${example}"
        bash run.sh
    )
done
