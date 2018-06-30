#include "stdafx.h"

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

	const double BASELINE = -(-3.745166) / 6.471884; // Distance between the two cameras
	const double FOCAL = 647.1884; // Focal Length in pixels

	Mat g1, g2, disp8;
	Mat g1_res, g2_res, disp8_res;

	Mat leftimg = imread("C:/Users/josha/Documents/Uni/2018/Project/2010_03_04_drive_0041/I1_000388.png");
	Mat rightimg = imread("C:/Users/josha/Documents/Uni/2018/Project/2010_03_04_drive_0041/I2_000388.png");

	Mat left_res = imread("C:/Users/josha/Pictures/I1_000163_res.png");
	Mat right_res = imread("C:/Users/josha/Pictures/I2_000163_res.png");

	if ((!leftimg.data) || (!rightimg.data) || (!left_res.data) || (!right_res.data))
	{
		return -1;
	}

	cvtColor(leftimg, g1, COLOR_BGR2GRAY);
	cvtColor(rightimg, g2, COLOR_BGR2GRAY);

	cvtColor(left_res, g1_res, COLOR_BGR2GRAY);
	cvtColor(right_res, g2_res, COLOR_BGR2GRAY);


	imshow("left", g1);
	imshow("right", g2);

	imshow("left resized", g1_res);
	imshow("right resized", g2_res);

	// Remove noise by blurring with a Gaussian filter
	GaussianBlur(g1, g1, Size(3, 3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(g2, g2, Size(3, 3), 0, 0, BORDER_DEFAULT);

	GaussianBlur(g1_res, g1_res, Size(3, 3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(g2_res, g2_res, Size(3, 3), 0, 0, BORDER_DEFAULT);

	Size imagesize = g1.size();
	Size imsize = g1_res.size();

	Mat disparity_left = Mat(imagesize.height, imagesize.width, CV_16S);
	Mat disparity_right = Mat(imagesize.height, imagesize.width, CV_16S);

	Mat disp_left_res = Mat(imsize.height, imsize.width, CV_16S);
	Mat disp_right_res = Mat(imsize.height, imsize.width, CV_16S);


	Ptr<StereoBM> sbm = StereoBM::create(256, 17);

	//sbm->setDisp12MaxDiff(1);
	//sbm->setSpeckleRange(8);
	//sbm->setSpeckleWindowSize(9);
	//sbm->setMinDisparity(-39);
	//sbm->setPreFilterCap(61);
	//sbm->setPreFilterSize(5);


	sbm->setUniquenessRatio(15);
	sbm->setTextureThreshold(0.0002);
	sbm->compute(g1, g2, disparity_left);

	Ptr<StereoBM> sbm_res = StereoBM::create(256, 9);

	sbm_res->setUniquenessRatio(15);
	sbm_res->setTextureThreshold(0.0002);
	sbm_res->compute(g1_res, g2_res, disp_left_res);

	normalize(disparity_left, disp8, 0, 255, CV_MINMAX, CV_8U);

	normalize(disp_left_res, disp8_res, 0, 255, CV_MINMAX, CV_8U);

	imshow("disp", disp8);
	imshow("disp resized", disp8_res);

	float test = 0;

	int y = 0;
	int x = 0;

	imwrite("C:/Users/josha/Pictures/disparity_far.png", disp8);
	//imwrite("C:/Users/josha/Pictures/disparity_res.png", disp8_res);

	cout << "pixel value: " << (int)disp8.at<unsigned char>(171,688) << endl;
	cout << "distance : " << (FOCAL*BASELINE) / (int)disp8.at<unsigned char>(171, 688) << " m" << endl;

	//for (y = 0; y < g1.rows; y++) {
	//	for (x = 0; x < g1.cols; x++) {
	//		if ((int)disp8.at<unsigned char>(y, x) >= 253) {
	//			cout << "Y: " << y << ", X: " << x << endl;
	//			cout << "Distance : " << (FOCAL*BASELINE) / (int) disp8.at<unsigned char>(y, x) << " m" << endl;
	//			break;
	//		}
	//	}
	//}

	
	waitKey(0);
}