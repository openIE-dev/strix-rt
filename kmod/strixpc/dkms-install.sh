#!/bin/sh
# Register strixpc with DKMS (rebuilt automatically for new kernels). Does not autoload at boot;
# to autoload: echo strixpc | sudo tee /etc/modules-load.d/strixpc.conf
# Rollback: sudo rmmod strixpc; sudo dkms remove strixpc/0.1.0 --all; sudo rm -rf /usr/src/strixpc-0.1.0
set -e
V=0.1.0; SRC=/usr/src/strixpc-$V
cd "$(dirname "$0")"
sudo rm -rf "$SRC"; sudo mkdir -p "$SRC"
sudo cp strixpc.c strixpc.h Makefile dkms.conf "$SRC"/
sudo dkms remove strixpc/$V --all >/dev/null 2>&1 || true
sudo dkms add strixpc/$V
sudo dkms build strixpc/$V
sudo dkms install strixpc/$V
dkms status strixpc
