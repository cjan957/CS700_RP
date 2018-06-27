#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

#define YML_LOCATION "vehicle_detector.yml"
#define TEST_VIDEO_LOCATION "april21.avi"
#define IMAGE_SIZE Size(64,64)
#define CONFIDENCE_THRESHOLD 0.5
#define BYPASS_CONFIDENCE_CHECK 1

vector<float> get_svm_detector(const Ptr<SVM> &svm);

int main()
{	 
	
	Mat img = imread("stereo_dataset/I1_000164.png");
	
	Mat blurred;
	
	GaussianBlur(img,blurred,Size(3,3),0,0, BORDER_DEFAULT);
	
	if(img.empty())
	{
		cerr << "unable to open" << endl;
	}
	
	
	clog << "Loading SVM file.. Please wait.. " << endl;
	Ptr<SVM> svm = StatModel::load<SVM>(YML_LOCATION); ;
	clog << "YML file loaded!" << endl;
	
	vector<float> hog_detector = get_svm_detector(svm);
	
	HOGDescriptor hog;
	hog.winSize = IMAGE_SIZE;
	hog.setSVMDetector(hog_detector);
	
	
	vector<Rect> detections;
	vector<double> foundWeights;
	
	hog.detectMultiScale(blurred, detections, foundWeights);
		
	double confidence;
	Scalar confidence_colour;
		
	for(size_t j = 0; j < detections.size(); j++)
	{
		confidence = foundWeights[j] * foundWeights[j];
		
		if((confidence > CONFIDENCE_THRESHOLD) || BYPASS_CONFIDENCE_CHECK)
		{
			confidence_colour = Scalar(0, confidence * 200, 0);
			rectangle(img, detections[j], confidence_colour, img.cols / 400 + 1);
		}
	}
		
	imshow("Vehicle Detection", img);
	
	if(waitKey(1) == 27)
	{
		return 1;
	}
	
	
	waitKey(0);
	
}

vector<float> get_svm_detector(const Ptr<SVM> &svm)
{
	//The method returns all the support vector as floating-point 
	//matrix, where support vectors are stored as matrix rows.
	Mat sv = svm->getSupportVectors();
	
	const int sv_total = sv.rows;
	
	//alpha: output vector for weights, corresponding to different
	//support vectors. For linear SVM = all 1
	//svidx: vector of indices of support vectors
	Mat alpha, svidx;
	
	//rho: scalar subtracted from the weighted sum of kernel responses
	double rho = svm->getDecisionFunction(0, alpha, svidx);
	
	//sanity checks
	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	
	
	vector<float> hog_detector(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
	
	return hog_detector;
}
