#!/bin/bash
python3 configure.py --prob=disk_snowline_2D_RT_erg_2_twopop --ndustfluid=5 --flux=hllc -omp -hdf5 --hdf5_path=${hdf5_path} -h5double --nghost=2 --coord=spherical_polar --eos=general/eos_phase_change --np=2 --nz=2 --cxx=g++
# python3 configure.py --prob=disk_snowline_2D_RT_erg_2 --ndustfluid=3 --np=1 --nz=2 --flux=hllc -omp -hdf5 -h5double --hdf5_path=${hdf5path} --nghost=2 --coord=spherical_polar --eos=general/eos_phase_change --cxx=g++

make clean
make -j 16
