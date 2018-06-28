#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <stdio.h>
#include <iostream>

#include <sys/types.h>
#include <dirent.h>


using namespace cv;
using namespace cv::ml;
using namespace std;

#define YML_LOCATION "vehicle_detector_filter.yml"
#define TEST_VIDEO_LOCATION "april21.avi"
#define IMAGE_SIZE Size(64,64)
#define CONFIDENCE_THRESHOLD 0.5
#define BYPASS_CONFIDENCE_CHECK 0

vector<float> get_svm_detector(const Ptr<SVM> &svm);

int main()
{	 
	
	clog << "Loading SVM file.. Please wait.. " << endl;
	Ptr<SVM> svm = StatModel::load<SVM>(YML_LOCATION); ;
	clog << "YML file loaded!" << endl;
	
	vector<float> hog_detector = get_svm_detector(svm);
		
	HOGDescriptor hog;
	hog.winSize = IMAGE_SIZE;
	hog.setSVMDetector(hog_detector);
	
	double confidence;
	Scalar confidence_colour;
		
	int fileCount;
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir ("/home/pi/Desktop/700/stereo_dataset/left")) != NULL) {
	/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {
		fileCount++;
	}
	closedir (dir);
	} else {
		/* could not open directory */
		perror ("");
		return EXIT_FAILURE;
	}
	
	cout << "LENGTH IS " << fileCount - 2 << endl;
	
	int i = 0;
	
	string openFile = "I1_000";
	
	string testFile = "";
	
	string direct= "/home/pi/Desktop/700/stereo_dataset/left/";
	
	Mat image;
	Mat blurred;
	Mat original;
	
	
	for (i = 0; i < fileCount - 2; i++) {
		
		if (i < 10)
			
			testFile = openFile + "00" + to_string(i) + ".png";
		
		else if ((i >= 10) && (i < 100)) {
			
			testFile = openFile + "0" + to_string(i) + ".png";
			
		} else {
			
			testFile = openFile + to_string(i) + ".png";		
			
		}
				
		image = imread(direct + testFile);
				
		GaussianBlur(image, blurred, Size(3,3), 0, 0, BORDER_DEFAULT);
		
		//cvtColor(blurred, blurred, CV_BGR2GRAY);

		vector<Rect> detections;
		vector<double> foundWeights;
		
		hog.detectMultiScale(blurred, detections, foundWeights);
		
		for(size_t i = 0; i < detections.size(); i++)
		{
			confidence = foundWeights[i] * foundWeights[i];
			if((confidence > CONFIDENCE_THRESHOLD) || BYPASS_CONFIDENCE_CHECK)
			{
				confidence_colour = Scalar(0, confidence * 200, 0);
				rectangle(image, detections[i], confidence_colour, image.cols / 400 + 1);
			}
		}
		
		imshow("Vehicle Detection", image);
		
		if(waitKey(30) == 27) //escape
		{ 
			return 1;
		}	
		
	}
	
	/*
	VideoCapture sequence("stereo_dataset/I1_%06d.png", CAP_IMAGES);
	if(!sequence.isOpened())
	{
		cerr << "error, cant open sequence \n" << endl;
		return 1;
	}
	*/
	
	
	//~ clog << "Loading SVM file.. Please wait.. " << endl;
	//~ Ptr<SVM> svm = StatModel::load<SVM>(YML_LOCATION); ;
	//~ clog << "YML file loaded!" << endl;
	
	//~ vector<float> hog_detector = get_svm_detector(svm);
	
	//~ HOGDescriptor hog;
	//~ hog.winSize = IMAGE_SIZE;
	//~ hog.setSVMDetector(hog_detector);
	
	//~ double confidence;
	//~ Scalar confidence_colour;
	
	//~ Mat image;
	//~ Mat blurred;
	//~ Mat original;
	
	
	//~ for(;;)
	//~ {
		
		//~ sequence >> image;
		//~ original = image;
		
		//~ if(image.empty())
		//~ {
			//~ cout << "Done" << endl;
		//~ }
		
		
		//~ GaussianBlur(image, blurred, Size(3,3), 0, 0, BORDER_DEFAULT);
		
		//~ //cvtColor(blurred, blurred, CV_BGR2GRAY);

		//~ vector<Rect> detections;
		//~ vector<double> foundWeights;
		
		//~ hog.detectMultiScale(blurred, detections, foundWeights);
		
		//~ for(size_t i = 0; i < detections.size(); i++)
		//~ {
			//~ confidence = foundWeights[i] * foundWeights[i];
			//~ if((confidence > CONFIDENCE_THRESHOLD) || BYPASS_CONFIDENCE_CHECK)
			//~ {
				//~ confidence_colour = Scalar(0, confidence * 200, 0);
				//~ rectangle(original, detections[i], confidence_colour, original.cols / 400 + 1);
			//~ }
		//~ }
		
		//~ imshow("Vehicle Detection", original);
		
		//~ if(waitKey(1) == 27) //escape
		//~ { 
			//~ return 1;
		//~ }
	//~ }		
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
