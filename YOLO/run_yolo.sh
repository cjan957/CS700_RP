echo Compiling...
g++ $(pkg-config --libs --cflags opencv) -o yolo yolo.cpp
echo Done!

./yolo