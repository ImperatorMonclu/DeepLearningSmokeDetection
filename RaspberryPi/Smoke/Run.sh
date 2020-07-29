#!bin/bash
source /root/Smoke/Settings.sh
python3.7 /root/Smoke/Run.py $GetData
if [ "$Loop" = true ]; then
    shutdown -h now
fi
