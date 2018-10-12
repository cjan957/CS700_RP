echo Compiling detect_vehicle_parallel...
g++ $(pkg-config --libs --cflags opencv) -o detect_vehicle_parallel detect_vehicle_parallel.cpp -pthread -std=c++11
echo Done!

./detect_vehicle_parallel