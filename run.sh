echo Compiling...
g++ $(pkg-config --libs --cflags opencv) -o detect_vehicle detect_vehicle.cpp
echo Done!

./detect_vehicle