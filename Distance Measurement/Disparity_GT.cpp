#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/calib3d.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"

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

Mat disp_GT, leftImg, rightImg, depthMap;

void pixelValue(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Pixel value at (" << x << "," << y << "): " << (float)disp_GT.at<uint16_t>(y, x) << endl;
		cout << "Disparity value at (" << x << "," << y << "): " << ((float)disp_GT.at<uint16_t>(y, x)) / 256 << endl;
		cout << "==========================================" << endl;
	}
}

cv::Mat DepthMap(cv::Mat &imageL, cv::Mat &imageR)
{

    /// DISPARITY MAP AND DEPTH MAP
    Mat left_for_matcher = imageL;
    Mat right_for_matcher = imageR;
    Mat left_disp,right_disp;
    Mat filtered_disp;
    int max_disp = 256; // n*16
    int wsize = 15;

	//~ max_disp/=2;
    //~ if(max_disp%16!=0)
    //~ {
		//~ max_disp += 16-(max_disp%16);
    //~ }
        
    // resize(imageL ,left_for_matcher ,Size(),0.5,0.5);
	// resize(imageR, right_for_matcher,Size(),0.5,0.5);


   // Perform matching and create the filter instance
   /* I am using StereoBM for faster processing. If speed is not critical, 
   though, StereoSGBM would provide better quality.
   The filter instance is created by providing the StereoMatcher instance
   that we intend to use. Another matcher instance is returned by the
   createRightMatcher function. These two matcher instances are then used
   to compute disparity maps both for the left and right views, that are
   required by the filter. */

    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;

    // BM
    cv::Ptr<cv::StereoBM >left_matcher = cv::StereoBM::create(max_disp,wsize);
    
    // SGBM
    //~ Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,wsize);
    //~ left_matcher->setMode(StereoSGBM::MODE_SGBM);
    
    wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
    cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

	left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
    right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);

    // Perform filtering
    /* Disparity maps computed by the respective matcher instances, as
    well as the source left view are passed to the filter. Note that we
    are using the original non-downscaled view to guide the filtering 
    process. 
    The disparity map is automatically upscaled in an edge-aware fashion
    to match the original view resolution. The result is stored in
    filtered_disp. */


    double lambda = 8000.0;   // hardcode
    double sigma = 2;       // hardcode
	int vis_mult = 2;

    //! [filtering]
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);

    wls_filter->filter(left_disp, imageL, filtered_disp, right_disp);

    cv::Mat raw_disp_vis;
    cv::ximgproc::getDisparityVis(left_disp,raw_disp_vis, vis_mult);
    cv::Mat filtered_disp_vis;
    cv::ximgproc::getDisparityVis(filtered_disp,filtered_disp_vis, vis_mult);


    return filtered_disp_vis;  // rerturning de depth map image.
}

int main()
{
	disp_GT = imread("/home/pi/Desktop/CS700_RP/stereo_dataset/disparity_GT/disp_occ/000172_10.png", IMREAD_ANYDEPTH);

	leftImg = imread("/home/pi/Desktop/CS700_RP/stereo_dataset/disparity_GT/image_0/000172_10.png", IMREAD_ANYDEPTH);
	rightImg = imread("/home/pi/Desktop/CS700_RP/stereo_dataset/disparity_GT/image_1/000172_10.png", IMREAD_ANYDEPTH);
	
	//~ imshow("LEFT" , leftImg);
	//~ imshow("RIGHT", rightImg);

	imshow("disp_GT", disp_GT);

	int x, y;
	x = 1001;
	y = 303;

	cout << "Pixel value at (" << x << "," << y << "): " << (float)disp_GT.at<uint16_t>(y,x) << endl;
	cout << "Disparity value at (" << x << "," << y << "): " << ((float) disp_GT.at<uint16_t>(y, x)) / 256 << endl;
	cout << "==========================================" << endl;

	setMouseCallback("disp_GT", pixelValue, NULL);
	
	Mat true_map;
	
	depthMap = DepthMap(leftImg, rightImg);
	
	//~ Mat orig_depthMap = Mat(leftImg.size().height, leftImg.size().width, CV_16S);
	
	//~ Ptr<StereoSGBM> disparity  = StereoSGBM::create(0,256,15);
	
	//~ disparity->compute(leftImg,rightImg,orig_depthMap);
					
	//~ orig_depthMap.convertTo(true_map, CV_32F, 1.0/16.0, 0.0);
	
	//~ cout << "Original Depth Map" << endl;
	//~ cout << "Pixel value at (" << x << "," << y << "): " << (float) orig_depthMap.at<uint16_t>(y,x) << endl;
	//~ cout << "Disparity value at (" << x << "," << y << "): " << ((float) orig_depthMap.at<uint16_t>(y, x)) / 256 << endl;
	//~ cout << "==========================================" << endl;
	
	//~ cout << "Converted Depth Map (32 bit float)" << endl;
	//~ cout << "Pixel value at (" << x << "," << y << "): " << true_map.at<float>(y,x) << endl;
	//~ cout << "Disparity value at (" << x << "," << y << "): " << (true_map.at<float>(y, x)) / 256 << endl;
	//~ cout << "==========================================" << endl;
	
	//~ imshow("True Calculated Depth Map", orig_depthMap);
	
	cout << "Filtered Depth Map" << endl;
	cout << "Pixel value at (" << x << "," << y << "): " << (float)depthMap.at<uint16_t>(y,x) << endl;
	cout << "Disparity value at (" << x << "," << y << "): " << ((float) depthMap.at<uint16_t>(y, x)) / 256 << endl;
	
	
	imshow("Depth Map", depthMap);
	
	

	waitKey(0);

}
