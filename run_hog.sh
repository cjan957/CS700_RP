echo Compiling hog-detect...
g++ $(pkg-config --libs --cflags opencv) -o HOG_performance HOG_Performance.cpp
echo Done!

./HOG_performance