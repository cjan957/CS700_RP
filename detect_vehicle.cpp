#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>


#include <string.h>
#include <string>
#include <utility>
#include <opencv2/calib3d.hpp>

#include <stdio.h>
#include <iostream>

#include <sys/types.h>
#include <dirent.h>

#include <typeinfo>

#include <iomanip>
#include <sstream>


using namespace cv;
using namespace cv::ml;
using namespace std;

#define YML_LOCATION "vehicle_detector_no_sobel.yml"
#define IMAGE_SIZE Size(64,64)

#define SEQUENCE_SET 1 // 1 for sequence of images, 0 for a single shot (include L and R), specify below
#define CAMERA_1_LOCATION "/home/pi/Desktop/700/stereo_dataset/resize_left/I1_000122.png"
#define CAMERA_2_LOCATION "/home/pi/Desktop/700/stereo_dataset/resize_right/I2_000122.png"
#define START_AT_SEQUENCE 100 // <- specify STARTING SEQUENCE HERE
#define CONFIDENCE_THRESHOLD 0.5 //specify confidence threshold for when by pass is 0
#define BYPASS_CONFIDENCE_CHECK 0 //draw black boxes when not confidence

#define LOOP 1



vector<float> get_svm_detector(const Ptr<SVM> &svm);
void CheckAndDraw(Mat &image, vector<Rect> &detections, vector<double> &foundWeights);
Point getPoints(Mat &image, vector<Rect> &detections, vector<double> &foundWeights);


float disparityMap(Mat imageL, Mat imageR, vector<Rect> &detections_L, vector<double> &foundWeights_L, vector<Rect> &detections_R, vector<double> &foundWeights_R);

int main()
{	 
	
	
	cout << setprecision(2) << fixed;
	
	clog << "Loading the Training file.. Please wait.. " << endl;
	Ptr<SVM> svm = StatModel::load<SVM>(YML_LOCATION);
	clog << "Training file loaded!" << endl;
	
	vector<float> hog_detector = get_svm_detector(svm);
		
	HOGDescriptor hog;
	hog.winSize = IMAGE_SIZE;
	hog.blockSize = Size(16,16);
	hog.cellSize = Size(8,8);
	
	hog.setSVMDetector(hog_detector);
	
	Mat image;
	Mat image_2;
	Mat blurred;
	Mat blurred_2;
	Mat original;
	Mat original_2;
	
	Mat ROI_L;
	Mat ROI_R;
	
	Mat disparity, disp8;
	Mat grayL;
	Mat grayR;
	Ptr<StereoBM> sbm = StereoBM::create(64, 21);
	
	Mat ROI_disp_L, ROI_disp_R;	
	
	vector<Rect> detections;
	vector<double> foundWeights;
	vector<Rect> detections_2;
	vector<double> foundWeights_2;

	float dist;
	Point pointLeft;
	Point pointRight;
	String dist_str;


#if SEQUENCE_SET	
		
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
	
	fileCount -= 2;
	
	cout << "LENGTH IS " << fileCount<< endl;
	
	int i = 0;
	
	string openFile = "I1_000";
	string openSecondFile = "I2_000";
	
	string testFile = "";
	string testFile_2 = "";
	
	string direct= "/home/pi/Desktop/700/stereo_dataset/resize_left/";
	string direct_right = "/home/pi/Desktop/700/stereo_dataset/resize_right/";

	//change starting image by changing i?
	for (i = START_AT_SEQUENCE; i < fileCount; i++) 
	{
				
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
	
		
		// Grayscale the image
		cvtColor(image, grayL, CV_BGR2GRAY);
		cvtColor(image_2, grayR, CV_BGR2GRAY);
		
		// Remove the noise
		GaussianBlur(grayL, blurred, Size(3,3), 0, 0, BORDER_DEFAULT);
		GaussianBlur(grayR, blurred_2, Size(3,3), 0, 0, BORDER_DEFAULT);
		
		// Limit the ROI (Region of Interest)
		Rect roi = Rect(0, 0, 392, blurred.size().height);
		ROI_L = Mat(blurred, roi);
		ROI_R = Mat(blurred_2, roi);
		
		
		hog.detectMultiScale(ROI_L, detections, foundWeights);
		hog.detectMultiScale(ROI_R, detections_2, foundWeights_2);
		
		disparity = Mat(grayL.size().height, grayL.size().width, CV_16S);
	
		sbm->setUniquenessRatio(15);
		sbm->setTextureThreshold(0.0002);
		sbm->compute(grayL, grayR, disparity);
	
		normalize(disparity, disp8, 0, 255, CV_MINMAX, CV_8U);
	
		imshow("Disparity Map - FULL", disp8);
		
		dist = disparityMap(ROI_L, ROI_R, detections, foundWeights, detections_2, foundWeights_2);
		
		stringstream stream;
		stream << fixed << setprecision(2) << dist;
		string s = stream.str();
		dist_str = "Distance: " + s + " m";
	
		pointLeft = getPoints(ROI_L, detections, foundWeights);
		pointRight = getPoints(ROI_R, detections_2, foundWeights_2);
	
		putText(image, dist_str, pointLeft, FONT_HERSHEY_PLAIN,1, Scalar(255,255,255),1,CV_AA);
		putText(image_2, dist_str, pointRight, FONT_HERSHEY_PLAIN,1, Scalar(255,255,255),1,CV_AA);
		
		
		CheckAndDraw(image, detections, foundWeights);
		CheckAndDraw(image_2, detections_2, foundWeights_2);
		
		imshow("L Vehicle Detection (Sequence)", image);
		imshow("R Vehicle Detection 2 (Sequence)", image_2);
		
		cout << "Image : " << i << endl;
		
		#if LOOP
			// Loop back the video
			if(i == fileCount - 1)
			{
				i = 0;
			}
		#endif	
		
		//~ if (waitKey(999999) == 37)
		//~ {
			//~ continue;
		//~ }
		
		if(waitKey(30) == 27) //escape
		{ 
			return 1;
		}
		

		
	}
	
#else
	
	const double BASELINE = -(-3.745166) / 6.471884; // Distance between the two cameras
	const double FOCAL = 647.1884; // Focal Length in pixels
	
	image = imread(CAMERA_1_LOCATION);
	image_2 = imread(CAMERA_2_LOCATION);
		
	// ------------ DISPARITY -------------
	// Grayscale the image
	cvtColor(image, grayL, CV_BGR2GRAY);
	cvtColor(image_2, grayR, CV_BGR2GRAY);
	
	// Remove the noise
	GaussianBlur(grayL, grayL, Size(3,3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(grayR, grayR, Size(3,3), 0, 0, BORDER_DEFAULT);
	
	disparity = Mat(grayL.size().height, grayL.size().width, CV_16S);
	
	sbm->setUniquenessRatio(15);
	sbm->setTextureThreshold(0.0002);
	sbm->compute(grayL, grayR, disparity);
	
	normalize(disparity, disp8, 0, 255, CV_MINMAX, CV_8U);
	
	imshow("Disparity Map", disp8);
	
	cout << "pixel value (original): " << (int)disp8.at<unsigned char>(110,327) << endl;
	cout << "distance : " << (FOCAL*BASELINE) / (int)disp8.at<unsigned char>(110,327) << " m" << endl;
	
	
	// ------- NORMAL OBJECT DETECTION
	// Remove the noise
	GaussianBlur(image, blurred, Size(3,3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(image_2, blurred_2, Size(3,3), 0, 0, BORDER_DEFAULT);
	
		
	hog.detectMultiScale(blurred, detections, foundWeights);
	hog.detectMultiScale(blurred_2, detections_2, foundWeights_2);
		
	//-----------------------------------------------------//
	
	dist = disparityMap(grayL, grayR, detections, foundWeights, detections_2, foundWeights_2);
		
	stringstream stream;
	
	stream << fixed << setprecision(2) << dist;
	
	string s = stream.str();
	
	String dist_str = "Distance: " + s + " m";
	
	cout << "------------- LEFT -------------" << endl;
	pointLeft = getPoints(image, detections, foundWeights);
	
	cout << "------------- RIGHT -------------" << endl;
	pointRight = getPoints(image_2, detections_2, foundWeights_2);
	
	putText(image, dist_str, pointLeft, FONT_HERSHEY_PLAIN,1, Scalar(255,255,255),1,CV_AA);
	putText(image_2, dist_str, pointRight, FONT_HERSHEY_PLAIN,1, Scalar(255,255,255),1,CV_AA);

			
	CheckAndDraw(image, detections, foundWeights);
	CheckAndDraw(image_2, detections_2, foundWeights_2);
	
	imshow("L Vehicle Detection (Static)", image);
	imshow("R Vehicle Detection 2 (Static)", image_2);
		
	if(waitKey(0) == 27) //escape
	{ 
		return 1;
	}	

#endif
}

float disparityMap(Mat imageL, Mat imageR, vector<Rect> &detections_L, vector<double> &foundWeights_L, vector<Rect> &detections_R, vector<double> &foundWeights_R)
{

	const double BASELINE = -(-3.745166) / 6.471884; // Distance between the two cameras
	const double FOCAL = 647.1884; // Focal Length in pixels
	float final_dist = 0;
	
	Ptr<StereoBM> sbm_crop = StereoBM::create(16, 7);
	
	Mat disparity, disp8;	
	Mat ROI_disp_L, ROI_disp_R;	

	Point pointLeft;
	Point pointRight;
	
	Rect roi_L, roi_R;

	double confidence;	
	
	// Left Image
	for(size_t i = 0; i < detections_L.size(); i++)
	{
		confidence = foundWeights_L[i] * foundWeights_L[i];
		if((confidence > CONFIDENCE_THRESHOLD) || BYPASS_CONFIDENCE_CHECK)
		{
			pointLeft = Point(detections_L[i].x, detections_L[i].y);
		}
	}
	
	// Right Image
	for(size_t i = 0; i < detections_R.size(); i++)
	{
		confidence = foundWeights_R[i] * foundWeights_R[i];
		if((confidence > CONFIDENCE_THRESHOLD) || BYPASS_CONFIDENCE_CHECK)
		{
			pointRight = Point(detections_R[i].x, detections_R[i].y);
		}
	}
	
	
	if ((pointLeft.x != 0) && (pointRight.x != 0)) 
	{
			
		roi_L = Rect(pointLeft.x -5 ,pointLeft.y,105,105);
		ROI_disp_L = Mat(imageL, roi_L);
	
		roi_R = Rect(pointRight.x - 5,pointLeft.y,105,105);
		ROI_disp_R = Mat(imageR, roi_R);
	
		imshow("LEFT IMAGE", ROI_disp_L);
		imshow("RIGHT IMAGE", ROI_disp_R);
	
		disparity = Mat(ROI_disp_L.size().height, ROI_disp_L.size().width, CV_16S);
	
		sbm_crop->setUniquenessRatio(15);
		sbm_crop->setTextureThreshold(0.0002);
		sbm_crop->compute(ROI_disp_L, ROI_disp_R, disparity);
	
		normalize(disparity, disp8, 0, 255, CV_MINMAX, CV_8U);
	
		imshow("DISPARITY", disp8);
	
		cout << "HERE" << endl;
		cout << "pixel value: " << (int)disp8.at<unsigned char>(60, 60) << endl;
		final_dist = (FOCAL*BASELINE) / (int)disp8.at<unsigned char>(60, 60);
		cout << "distance : " << final_dist << " m" << endl;
	}
	
	return final_dist;
	
	
	
}

Point getPoints(Mat &image, vector<Rect> &detections, vector<double> &foundWeights)
{
	double confidence;	
		
	Point point0;
	
	for(size_t i = 0; i < detections.size(); i++)
	{
		
		confidence = foundWeights[i] * foundWeights[i];
		if((confidence > CONFIDENCE_THRESHOLD) || BYPASS_CONFIDENCE_CHECK)
		{			
			//cout << "WITHIN THRESHOLD " << detections[i] << endl;
			point0 = Point(detections[i].x, detections[i].y + 105);
		} else {
			//cout << detections[i] << endl;
		}
	}
	
	return point0;
}


void CheckAndDraw(Mat &image, vector<Rect> &detections, vector<double> &foundWeights)
{
	double confidence;	
	Scalar confidence_colour;
		
	for(size_t i = 0; i < detections.size(); i++)
	{
		
		confidence = foundWeights[i] * foundWeights[i];
		if((confidence > CONFIDENCE_THRESHOLD) || BYPASS_CONFIDENCE_CHECK)
		{			
			confidence_colour = Scalar(0, confidence * 200, 0);
			rectangle(image, detections[i], confidence_colour, image.cols / 400 + 1);			
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
