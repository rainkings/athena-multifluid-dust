#!/bin/bash
# python3 configure.py --prob=disk_snowline_2D_RT_erg_2_twopop --ndustfluid=7 --flux=hllc -omp -hdf5 --hdf5_path=${hdf5_path} -h5double --nghost=2 --coord=spherical_polar --eos=general/eos_phase_change --np=2 --nz=2 --cxx=g++
# pure gas run
# python3 configure.py --prob=disk_snowline_2D_RT_erg_2_twopop --flux=hllc -omp -hdf5 --hdf5_path=${hdf5_path} -h5double --nghost=2 --coord=spherical_polar --eos=general/ideal --cxx=g++
# 1p 2z run
python3 configure.py --prob=disk_snowline_2D_RT_erg_2_twopop --ndustfluid=4 --flux=hllc -hdf5 -omp --hdf5_path=${hdf5_path} -h5double --nghost=2 --coord=spherical_polar --eos=general/eos_phase_change --np=1 --nz=2 --cxx=g++
# 2p 1z run
# python3 configure.py --prob=disk_snowline_2D_RT_erg_2_twopop --ndustfluid=5 --flux=hllc -hdf5 -omp --hdf5_path=${hdf5_path} -h5double --nghost=2 --coord=spherical_polar --eos=general/eos_phase_change --np=2 --nz=1 --cxx=g++
# single core run
# python3 configure.py --prob=disk_snowline_2D_RT_erg_2_twopop --ndustfluid=3 --flux=hllc -hdf5 --hdf5_path=${hdf5_path} -h5double --nghost=2 --coord=spherical_polar --eos=general/eos_phase_change --np=1 --nz=2 --cxx=g++
# python3 configure.py --prob=disk_snowline_2D_RT_erg_2_twopop --ndustfluid=3 --np=1 --nz=2 --flux=hllc -omp -hdf5 -h5double --hdf5_path=${hdf5path} --nghost=2 --coord=spherical_polar --eos=general/eos_phase_change --cxx=g++

# python3 configure.py --prob=disk_snowline_1D_R_erg_2_twopop --ndustfluid=3 --flux=hllc -hdf5 --hdf5_path=${hdf5_path} -h5double --nghost=2 --coord=cylindrical --eos=general/eos_phase_change --np=2 --nz=1 --cxx=g++
# python3 configure.py --prob=disk_snowline_1d_2p --ndustfluid=5 --flux=hllc -hdf5 --hdf5_path=${hdf5_path} -h5double --nghost=2 --coord=cylindrical --eos=general/eos_phase_change --np=2 --nz=1 --cxx=g++
# 1d 2z 2p run
# python3 configure.py --prob=disk_snowline_1d_2p --ndustfluid=7 --flux=hllc -hdf5 --hdf5_path=${hdf5_path} -h5double --nghost=2 --coord=cylindrical --eos=general/eos_phase_change --np=2 --nz=2 --cxx=g++
# 1d 1z 3p run
# python3 configure.py --prob=disk_snowline_1d_2p --ndustfluid=7 --flux=hllc -hdf5 --hdf5_path=${hdf5_path} -h5double --nghost=2 --coord=cylindrical --eos=general/eos_phase_change --np=3 --nz=1 --cxx=g++
# 1d 2z 3p run
# python3 configure.py --prob=disk_snowline_1d_2p --ndustfluid=10 --flux=hllc -hdf5 --hdf5_path=${hdf5_path} -h5double --nghost=2 --coord=cylindrical --eos=general/eos_phase_change --np=3 --nz=2 --cxx=g++

# python3 configure.py --prob=disk_snowline_1D_hydro --ndustfluid=3 --flux=hllc -hdf5 --hdf5_path=${hdf5_path} -h5double --nghost=2 --coord=cylindrical --eos=general/eos_phase_change --np=2 --nz=1 --cxx=g++
# python3 configure.py --prob=disk_snowline_1D_hydro --ndustfluid=3 --flux=hllc -hdf5 --hdf5_path=${hdf5path} -h5double --nghost=2 --coord=cylindrical --eos=general/phase_change --cxx=g++
make clean
make -j 16
