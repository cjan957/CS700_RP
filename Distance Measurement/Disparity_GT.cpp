#include "stdafx.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <algorithm>
#include <cstdlib>

#include <iomanip>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <cstdint>

using namespace std;
using namespace cv;

Mat disp;

void pixelValue(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Pixel value at (" << x << "," << y << "): " << (float)disp.at<uint16_t>(y, x) << endl;
		cout << "Disparity value at (" << x << "," << y << "): " << ((float)disp.at<uint16_t>(y, x)) / 256 << endl;
		cout << "==========================================" << endl;
	}
}

int main()
{
	disp = imread("C:/Users/Josh/Documents/Uni/Part4-Project/Disparity_Groundtruth/data_stereo_flow/training/disp_occ/000172_10.png", IMREAD_ANYDEPTH);

	imshow("disp_GT", disp);

	int x, y;
	x = 1001;
	y = 303;

	cout << "Pixel value at (" << x << "," << y << "): " << (float)disp.at<uint16_t>(y,x) << endl;
	cout << "Disparity value at (" << x << "," << y << "): " << ((float) disp.at<uint16_t>(y, x)) / 256 << endl;
	cout << "==========================================" << endl;

	setMouseCallback("disp_GT", pixelValue, NULL);

	waitKey(0);

}