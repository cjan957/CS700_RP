echo Compiling...
g++ $(pkg-config --libs --cflags opencv) -o detect_vehicle_parallel detect_vehicle_parallel.cpp
echo Done!

./detect_vehicle_parallel