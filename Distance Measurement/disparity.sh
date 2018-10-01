echo Compiling..

g++ $(pkg-config --libs --cflags opencv) -o disp_EVAL Disparity_EVAL.cpp

./disp_EVAL