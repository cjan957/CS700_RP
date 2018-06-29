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

#define YML_LOCATION "vehicle_detector_no_sobel.yml"
#define TEST_VIDEO_LOCATION "april21.avi"
#define IMAGE_SIZE Size(64,64)
#define CONFIDENCE_THRESHOLD 0.5
#define BYPASS_CONFIDENCE_CHECK 0

vector<float> get_svm_detector(const Ptr<SVM> &svm);

int main()
{	 
	
	clog << "Loading the Training file.. Please wait.. " << endl;
	Ptr<SVM> svm = StatModel::load<SVM>(YML_LOCATION); ;
	clog << "Training file loaded!" << endl;
	
	vector<float> hog_detector = get_svm_detector(svm);
		
	HOGDescriptor hog;
	hog.winSize = IMAGE_SIZE;
	hog.blockSize = Size(16,16);
	hog.cellSize = Size(8,8);
	
	hog.setSVMDetector(hog_detector);
	
	double confidence;
	double confidence_2;
	
	Scalar confidence_colour;
	Scalar confidence_colour_2;

		
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
	string openSecondFile = "I2_000";
	
	string testFile = "";
	string testFile_2 = "";
	
	string direct= "/home/pi/Desktop/700/stereo_dataset/resize_left/";
	string direct_right = "/home/pi/Desktop/700/stereo_dataset/resize_right/";
	
	Mat image;
	Mat image_2;
	Mat blurred;
	Mat blurred_2;
	Mat original;
	Mat original_2;
	
	//change starting image by changing i?
	for (i = 0; i < fileCount - 2; i++) {
		
		if(i >= fileCount - 3)
		{
			i = 0;
		}
		
		if (i < 10)
		{
			
			testFile = openFile + "00" + to_string(i) + ".png";
			testFile_2 = openSecondFile + "00" + to_string(i) + ".png";
		}
		else if ((i >= 10) && (i < 100)) 
		{
			
			testFile = openFile + "0" + to_string(i) + ".png";
			testFile_2 = openSecondFile + "0" + to_string(i) + ".png";
			
		} 
		else 
		{
			
			testFile = openFile + to_string(i) + ".png";	
			testFile_2 = openSecondFile + to_string(i) + ".png";	
			
		}
				
		image = imread(direct + testFile);
		image_2 = imread(direct_right + testFile_2);
		
				
		GaussianBlur(image, blurred, Size(3,3), 0, 0, BORDER_DEFAULT);
		GaussianBlur(image_2, blurred_2, Size(3,3), 0, 0, BORDER_DEFAULT);
		
		//cvtColor(blurred, blurred, CV_BGR2GRAY);

		vector<Rect> detections;
		vector<double> foundWeights;
		vector<Rect> detections_2;
		vector<double> foundWeights_2;
		
	
		hog.detectMultiScale(blurred, detections, foundWeights);
		hog.detectMultiScale(blurred_2, detections_2, foundWeights_2);
		
		for(size_t i = 0; i < detections.size(); i++)
		{
			confidence = foundWeights[i] * foundWeights[i];
			if((confidence > CONFIDENCE_THRESHOLD) || BYPASS_CONFIDENCE_CHECK)
			{
				confidence_colour = Scalar(0, confidence * 200, 0);
				rectangle(image, detections[i], confidence_colour, image.cols / 400 + 1);
			}
		}
		
		for(size_t i = 0; i < detections_2.size(); i++)
		{
			confidence_2 = foundWeights_2[i] * foundWeights_2[i];
			if((confidence_2 > CONFIDENCE_THRESHOLD) || BYPASS_CONFIDENCE_CHECK)
			{
				confidence_colour_2 = Scalar(0, confidence_2 * 200, 0);
				rectangle(image_2, detections_2[i], confidence_colour_2, image_2.cols / 400 + 1);
			}
		}
		
		
		imshow("Vehicle Detection", image);
		imshow("Vehicle Detection 2", image_2);
		
		if(waitKey(30) == 27) //escape
		{ 
			return 1;
		}	
		
	}
	
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
