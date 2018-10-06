echo Compiling..

g++ $(pkg-config --libs --cflags opencv) -o disp_IMAGE Disparity_Images.cpp

./disp_IMAGE