#!bin/bash
python3.7 /root/Smoke/Test.py
source /root/Smoke/Settings.sh
mount /dev/sda1 /media/usb
cp -r /root/Smoke/Test /media/usb
cp -r /root/Smoke/TestCam /media/usb
if [ "$GetData" = true ]; then
    cp -r /root/Smoke/Images /media/usb
    rm /root/Smoke/Images/smoke/*
    rm /root/Smoke/Images/neutral/*
fi
umount /media/usb
echo Loop = $Loop
echo GetData = $GetData
