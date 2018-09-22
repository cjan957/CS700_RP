echo Compiling..

g++ $(pkg-config --libs --cflags opencv) -o disp_GT Disparity_GT.cpp

./disp_GT