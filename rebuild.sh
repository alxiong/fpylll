#!/usr/bin/env bash

# Rebuild the ./fpylll-fplll/ cpp code and ./src/fpylll packages
# Copied from ./bootstrap.sh

jobs="-j 8 "
if [ "$1" = "-j" ]; then
   jobs="-j $2 "
fi

source ./activate

# Rebuild fplll first
cd fpylll-fplll || exit
make clean
cmake -DMAKE_INSTALL_PREFIX=$VIRTUAL_ENV ..
make $jobs
make install
cd ../..

# Rebuild fpylll package
$PIP install .
