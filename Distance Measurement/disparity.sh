g++ $(pkg-config --libs --cflags opencv) -o ROI_disp ROI_disparity.cpp

./ROI_disp