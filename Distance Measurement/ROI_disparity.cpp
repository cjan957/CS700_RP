//#include "stdafx.h"

// Disparity Map
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <string>
#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

int main()
{

	clog << "OPENS" << endl;
	
	const double BASELINE = -(-3.745166) / 6.471884; // Distance between the two cameras
	const double FOCAL = 647.1884; // Focal Length in pixels

	Mat g1, g2, disp8, disp8_crop;
	Mat smallLeft, smallRight;

	Mat leftimg = imread("/home/pi/Desktop/700/stereo_dataset/left/I1_000153.png");
	Mat rightimg = imread("/home/pi/Desktop/700/stereo_dataset/right/I2_000153.png");

	if ((!leftimg.data) || (!rightimg.data))
	{
		cout << "FAILED" << endl;
		return -1;
	}

	cvtColor(leftimg, g1, COLOR_BGR2GRAY);
	cvtColor(rightimg, g2, COLOR_BGR2GRAY);

	imshow("left", g1);
	//imshow("right", g2);

	// Remove noise by blurring with a Gaussian filter
	GaussianBlur(g1, g1, Size(3, 3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(g2, g2, Size(3, 3), 0, 0, BORDER_DEFAULT);

	Size imagesize = g1.size();
	Mat disparity_left = Mat(imagesize.height, imagesize.width, CV_16S);
	Mat disparity_right = Mat(imagesize.height, imagesize.width, CV_16S);

	// (x,y, image width,image height)
	Rect roi_L = Rect(607, 146, 100, 100);
	smallLeft = Mat(g1, roi_L);

	Rect roi_R = Rect(560, 150, 100, 100);
	smallRight = Mat(g2, roi_R);
	
	imshow("Test_L", smallLeft);
	imshow("Test_R", smallRight);

	imagesize = smallLeft.size();
	Mat disp_cropped = Mat(imagesize.height, imagesize.width, CV_16S);

	Ptr<StereoBM> sbm = StereoBM::create(256, 17);

	Ptr<StereoBM> sbm_crop = StereoBM::create(32, 11);

	sbm->setUniquenessRatio(15);
	sbm->setTextureThreshold(0.0002);
	sbm->compute(g1, g2, disparity_left);

	sbm_crop->setUniquenessRatio(15);
	sbm_crop->setTextureThreshold(0.0002);
	sbm_crop->compute(smallLeft, smallRight, disp_cropped);

	normalize(disparity_left, disp8, 0, 255, CV_MINMAX, CV_8U);
	normalize(disp_cropped, disp8_crop, 0, 255, CV_MINMAX, CV_8U);

	//imwrite("C:/Users/josha/Pictures/disp_test.png", disp8);

	//cout << disp8_crop << endl;

	imshow("disp", disp8);
	imshow("disp_crop", disp8_crop);

	cout << "pixel value: " << (int)disp8_crop.at<unsigned char>(64, 85) << endl;
	cout << "distance : " << (FOCAL*BASELINE) / (int)disp8_crop.at<unsigned char>(64, 85) << " m\n" << endl;
	
	cout << "========== ORIGINAL =================" << endl;

	cout << "pixel value: " << (int)disp8.at<unsigned char>(220, 688) << endl;
	cout << "distance : " << (FOCAL*BASELINE) / (int)disp8.at<unsigned char>(220, 688) << " m" << endl;

	//(int)disp8.at<unsigned char>(y, x)

	waitKey(0);
}
