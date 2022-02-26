#!/bin/bash

n=$1
T=$2
NStride=$3
TStride=$4
fn=times_${n}_${T}_${NStride}_${TStrade}.dat

EXEC=../../build/examples/solver_timer
PLOTTER=../python/plot_times.py

[ ! -f "$fn" ] && rm -f ${fn}

[ ! -f "$EXEC" ] && echo "$EXEC does not exist; just find it and set appropriately."
[ ! -f "PLOTTER" ] && echo "$PLOTTER does not exist; just find it and set appropriately."

# all distros should support
CPUInfo=`lscpu | grep -i "Model name" | cut -d : -f 2 | gawk '{$1=$1}; 1'`

echo "Writing raw runtimes to "${fn}

# Create raw timings across 
# n in {T, T+NStride, T+2*NStride,..., n}
# T in {TStride, 2*TStride,..., T}
eval $EXEC $n $T $NStride $TStride > $fn
python $PLOTTER $n $T $NStride $TStride "$CPUInfo" $fn
