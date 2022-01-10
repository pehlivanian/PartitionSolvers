#!/bin/bash

n=$1
T=$2
NStride=$3
TStride=$4
fn=times_${n}_${T}_${NStride}_${TStrade}.dat

# all distros should support
CPUInfo=`lscpu | grep -i "Model name" | cut -d : -f 2 | gawk '{$1=$1}; 1'`

echo "Writing raw runtimes to "${fn}

# Many T runs for fixed n
# ./build/bin/solver_timer $n $T $NStride $TStride > $fn
python plot_times.py $n $T $NStride $TStride "$CPUInfo" $fn