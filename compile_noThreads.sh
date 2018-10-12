echo Compiling detect_vehicle.cpp...
g++ $(pkg-config --libs --cflags opencv) -o detectVehicle_Old detect_vehicle.cpp
echo Done!

./detectVehicle_Old