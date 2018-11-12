source /opt/poky/2.1.3/environment-setup-aarch64-poky-linux
make clean; make -j$(nproc) YOCTO=1
sudo cp -rf darknet ocl $_NFS/darknet_yechan
