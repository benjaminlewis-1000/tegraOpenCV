#! /bin/bash

USER=`who | awk '{print $1}' | sort -u`

sudo apt-get install git

cd /home/$USER

pwd
git clone https://github.com/jetsonhacks/installGrinch.git
cd installGrinch
./installGrinch.sh
