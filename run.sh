g++ $(pkg-config --libs --cflags opencv) -o detect_vehicle detect_vehicle.cpp

./detect_vehicle