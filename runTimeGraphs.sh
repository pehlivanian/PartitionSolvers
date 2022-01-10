#!/bin/bash

n=$1
T=$2
NStride=$3
TStride=$4
fn=times_${n}_${T}_${NStride}_${TStrade}.dat

# all distros should support
CPUInfo=`lscpu | grep -i "Model name" | cut -d : -f 2 | gawk '{$1=$1}; 1'`

echo "Writing raw runtimes to "${fn}

# Create raw timings across 
# n in {T, T+NStride, T+2*NStride,..., n}
# T in {TStride, 2*TStride,..., T}
./build/bin/solver_timer $n $T $NStride $TStride > $fn
python plot_times.py $n $T $NStride $TStride "$CPUInfo" $fn