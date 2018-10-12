echo Compiling multiCamDetect...
g++ $(pkg-config --libs --cflags opencv) -o multipleCamDetection multipleCameraDetection.cpp
echo Done!

./multipleCamDetection