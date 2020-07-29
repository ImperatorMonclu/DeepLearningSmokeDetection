#!bin/bash
cd /
sudo apt-get update -y
sudo apt-get upgrade -y
sudo mkdir /media
sudo mkdir /media/usb
sudo chown -R root:root /media/usb
sudo mount /dev/sda1 /media/usb
sudo cp -r /media/usb/Smoke /root
sudo umount /media/usb
sudo apt-get install -y python3.7 libgeos-3.7.1 libgeos-c1v5 libatlas3-base libwebp6 libpng16-16 libjpeg62-turbo libtiff5 libjasper1 libilmbase23 libopenexr23 libavcodec58 libavformat58 libswscale5 libqtgui4 libqt4-test libhdf5-103
sudo mkdir /root/Smoke/Libraries
cd /root/Smoke/Libraries
sudo apt-get download -y python3-distutils
sudo dpkg-deb -x python3-distutils_3.7.3-1_all.deb /
sudo curl https://bootstrap.pypa.io/get-pip.py | python3.7
sudo wget https://www.piwheels.org/simple/numpy/numpy-1.18.4-cp37-cp37m-linux_armv7l.whl
sudo wget https://www.piwheels.org/simple/scipy/scipy-1.4.1-cp37-cp37m-linux_armv7l.whl
sudo wget https://www.piwheels.org/simple/opencv-python/opencv_python-4.1.1.26-cp37-cp37m-linux_armv7l.whl
sudo wget https://www.piwheels.org/simple/kiwisolver/kiwisolver-1.2.0-cp37-cp37m-linux_armv7l.whl
sudo wget https://www.piwheels.org/simple/matplotlib/matplotlib-3.2.1-cp37-cp37m-linux_armv7l.whl
sudo wget https://www.piwheels.org/simple/PyWavelets/PyWavelets-1.1.1-cp37-cp37m-linux_armv7l.whl
sudo wget https://www.piwheels.org/simple/Pillow/Pillow-7.1.2-cp37-cp37m-linux_armv7l.whl
sudo wget https://www.piwheels.org/simple/imagecodecs/imagecodecs-2020.2.18-cp37-cp37m-linux_armv7l.whl
sudo wget https://www.piwheels.org/simple/scikit-image/scikit_image-0.17.2-cp37-cp37m-linux_armv7l.whl
sudo wget https://www.piwheels.org/simple/grpcio/grpcio-1.29.0-cp37-cp37m-linux_armv7l.whl
sudo wget https://www.piwheels.org/simple/h5py/h5py-2.10.0-cp37-cp37m-linux_armv7l.whl
sudo wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.1.0/tensorflow-2.1.0-cp37-none-linux_armv7l.whl
sudo pip install --upgrade pip
sudo pip install setuptools
sudo pip install numpy-1.18.4-cp37-cp37m-linux_armv7l.whl
sudo pip install scipy-1.4.1-cp37-cp37m-linux_armv7l.whl
sudo pip install opencv_python-4.1.1.26-cp37-cp37m-linux_armv7l.whl
sudo pip install kiwisolver-1.2.0-cp37-cp37m-linux_armv7l.whl
sudo pip install matplotlib-3.2.1-cp37-cp37m-linux_armv7l.whl
sudo pip install PyWavelets-1.1.1-cp37-cp37m-linux_armv7l.whl
sudo pip install Pillow-7.1.2-cp37-cp37m-linux_armv7l.whl
sudo pip install imagecodecs-2020.2.18-cp37-cp37m-linux_armv7l.whl
sudo pip install scikit_image-0.17.2-cp37-cp37m-linux_armv7l.whl
sudo pip install imgaug
sudo pip install grpcio-1.29.0-cp37-cp37m-linux_armv7l.whl
sudo pip install h5py-2.10.0-cp37-cp37m-linux_armv7l.whl
sudo pip install tensorflow-2.1.0-cp37-none-linux_armv7l.whl
cd /
sudo echo "export LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1" >>/root/.bashrc
source /root/.bashrc
sudo echo "sudo bash /root/Smoke/Run.sh" >>/etc/profile
sudo rm -r /root/Smoke/Libraries
python3.7 /root/Smoke/Run.py
python3.7 /root/Smoke/Test.py
source /root/Smoke/Settings.sh
sudo echo Loop = $Loop
sudo echo GetData = $GetData
