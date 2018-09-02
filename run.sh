echo Compiling singleDetection...
g++ $(pkg-config --libs --cflags opencv) -o singleDetection singleDetection.cpp
echo Done!

./singleDetection