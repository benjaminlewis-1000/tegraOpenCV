# /bin/bash

# Some general development libraries
sudo apt-get install build-essential make cmake cmake-curses-gui g++ -y
# libav video input/output development libraries
sudo apt-get install libavformat-dev libavutil-dev libswscale-dev -y
# Video4Linux camera development libraries
sudo apt-get install libv4l-dev -y 
# Eigen3 math development libraries
sudo apt-get install libeigen3-dev -y
# OpenGL development libraries (to allow creating graphical windows)
sudo apt-get install libglew1.6-dev -y
# GTK development libraries (to allow creating graphical windows)
sudo apt-get install libgtk2.0-dev -y

USER=`who | awk '{print $1}' | sort -u`

cd /home/$USER/Downloads
wget -O opencv-2.4.11.zip https://codeload.github.com/Itseez/opencv/zip/2.4.11

unzip opencv-2.4.11.zip
mv opencv-2.4.11 /home/$USER
cd /home/$USER/opencv-2.4.11
mkdir build
cd build
cmake -DWITH_CUDA=ON -DCUDA_ARCH_BIN="3.2" -DCUDA_ARCH_PTX="" -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DWITH_V4L=ON -DWITH_QT=ON -DWITH_OPENGL=ON -DWITH_VTK=ON .. 
sudo make -j8 install

# Test out with a few test files

cd ../samples/cpp
g++ edge.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -o edge
./edge

g++ laplace.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videostab -o laplace
./laplace

cd ../gpu
g++ houghlines.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videostab -o houghlines
./houghlines ../cpp/logo_in_clutter.png


