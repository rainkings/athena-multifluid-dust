#!/bin/bash
python3 configure.py --prob=disk_snowline_2D_RT_erg_2_twopop --ndustfluid=3 --flux=hllc -omp -hdf5 --hdf5_path=${hdf5_path} -h5double --nghost=2 --coord=spherical_polar --eos=general/eos_phase_change --cxx=g++

make clean
make -j 8
