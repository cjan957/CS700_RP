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

#include <string.h>
#include <string>
#include <utility>

#include <stdio.h>
#include <iostream>

#include <sys/types.h>
#include <dirent.h>

#include <typeinfo>

#include <iomanip>
#include <sstream>

#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <asm/unistd.h>


using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace cv::xfeatures2d;

//Location of files
#define YML_LOCATION "/home/pi/Desktop/CS700_RP/YML/vehicle_detector_new.yml"
#define L_CAMERA_SRC_DIR "/home/pi/Desktop/CS700_RP/stereo_dataset/city/left/"
#define R_CAMERA_SRC_DIR "/home/pi/Desktop/CS700_RP/stereo_dataset/city/right/"

#define DEBUG 0

//HOG configs
#define HOG_IMAGE_SIZE Size(64,64)
#define HOG_BLOCK_SIZE Size(16,16)
#define HOG_CELL_SIZE Size(8,8)

//STEREO configs
// cv::StereoBM::create(X,Y)
#define STEREO_DISPARITY_SEARCH_RANGE 64
#define STEREO_BLOCK_SIZE 31

//SBM & Disparity
#define UNIQUERATIO 15
#define TEXTURETHRESHOLD 0.0002

//Preprocessing configs
#define GAUSSIAN_KERNEL_SIZE Size(3,3)

//Starting image sequence
#define IMG_STARTING_SEQUENCE 10
#define IMG_STOPPING_SEQUENCE 60 

//Settings
#define CONFIDENCE_THRESHOLD 0.7
#define BYPASS_CONFIDENCE_CHECK 0

//ROI, cropping x and y
#define ROI_X 
#define ROI_Y

//Looping and Debugging
#define LOOP_IMAGES 0
#define PRESS_NEXT 0

struct CorrespondingPoint
{
	Point leftPoint;
	Point rightPoint;
}tempPoints;


//variables
long long inst_count;
int inst_fd;
vector<CorrespondingPoint> points;

//functions declarations
Point getPoints(Mat &image, vector<Rect> &detections, vector<double> &foundWeights);
vector<float> get_svm_detector(const Ptr<SVM> &svm);
vector<Rect> HOGConfidenceFilter(vector<Rect> &detections, vector<double> &foundWeights);

Ptr<SVM> LoadTrainingFile();
int FileCounter();
void SetupHOG(HOGDescriptor &hog, Ptr<SVM> svm);

void PreProcessing(Mat &imageL, Mat &imageR);
void FileNameDetermine(int order, String &fileName_L, String &fileName_R);

float disparityMap(Mat imageL, Mat imageR, Point pointLeft, Point pointRight, int index);
float disparityMap(Mat imageL, Mat imageR, vector<CorrespondingPoint> points);

void PointMatcher(Mat imageL, Mat imageR, vector<Rect>&detections_L, vector<Rect>&detections_R);
void CheckAndDraw(Mat &image, vector<Rect> &detections, vector<double> &foundWeights);
void HOGConfidenceFilter(vector<Rect> &detections, vector<double> &foundWeights, vector<Rect> &new_detections, vector<double> &new_foundWeights);

static void setup_counters(void);
static void stop_counters(void);
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, 
int cpu, int group_fd, unsigned long flags);

Mat grayL, grayR;

int main()
{
	
	//setNumThreads(0); //force to 1 core

	Ptr<SVM> svm;
	svm = LoadTrainingFile();
	if(!svm)
	{
		//loading failed
		return 0;
	}
	
	cout << "Press Enter to continue" << endl;
	cin.get();
		
	//Prepare HOG Descriptor to detect vehicles
	HOGDescriptor hog;
	SetupHOG(hog, svm);
	
	//Prepare OpenCV variables to work on images
	Mat image_L, image_R;
	Mat original_image_L, original_image_R;
	Mat ROI_L, ROI_R;
	Mat ROI_disp_L, ROI_disp_R;
	Mat disparity, disp8;
	

	//StereoBM
	Ptr<StereoBM> sbm; 
	sbm = StereoBM::create(STEREO_DISPARITY_SEARCH_RANGE, STEREO_BLOCK_SIZE);
	sbm->setUniquenessRatio(UNIQUERATIO);
	sbm->setTextureThreshold(TEXTURETHRESHOLD);
	
	vector<Rect> detections_L, detections_R;
	vector<double> weights_L, weights_R;
	
	vector<Rect> filteredDetections_L, filteredDetections_R;
	vector<double> filteredWeights_L, filteredWeights_R;
	
	float dist;
	String dist_str;
	Point pointLeft;
	Point pointRight;
	
	int fileCount = FileCounter();
	
	//setup_counters();
	
	String fileName_L, fileName_R;
	
	time_t start, end;
	time(&start);
	
	
	
	for (int i = IMG_STARTING_SEQUENCE; i <= IMG_STOPPING_SEQUENCE; i++)
	{
		cout << i << endl;
		FileNameDetermine(i, fileName_L, fileName_R);
		
		image_L = imread(L_CAMERA_SRC_DIR + fileName_L);
		image_R = imread(R_CAMERA_SRC_DIR + fileName_R);
		
		
		original_image_L = image_L;
		original_image_R = image_R;
		
		PreProcessing(image_L, image_R);
		
		//crop the image by ROI
		//Rect roi = Rect(0,0, ROI_X, ROI_Y);
		Rect roi = Rect(0, 0, image_L.size().width, image_L.size().height);
		
		image_L = Mat(image_L, roi);
		image_R = Mat(image_R, roi);
		
		
		hog.detectMultiScale(image_L, detections_L, weights_L);
		hog.detectMultiScale(image_R, detections_R, weights_R);
		
		filteredDetections_L.clear();
		filteredDetections_R.clear();
		filteredWeights_L.clear();
		filteredWeights_R.clear();
		
		HOGConfidenceFilter(detections_L, weights_L, filteredDetections_L, filteredWeights_L);
		HOGConfidenceFilter(detections_R, weights_R, filteredDetections_R, filteredWeights_R);
		
		PointMatcher(image_L, image_R, filteredDetections_L, filteredDetections_R);
		
		// Disparity Map		
		disparity = Mat(grayL.size().height, grayR.size().width, CV_16S);
		
		sbm->setUniquenessRatio(15);
		sbm->setTextureThreshold(0.0002);
		sbm->compute(grayL, grayR, disparity);
		
		normalize(disparity, disp8, 0, 255, CV_MINMAX, CV_8U);
		
		#if DEBUG
		imshow("Disparity Map - FULL", disp8);
		#endif
		
		stringstream stream;
		string s;
		
		for (int z = 0; z < points.size() ; z++) 
		{

			pointLeft = points.at(z).leftPoint;
			pointRight = points.at(z).rightPoint;
				
			dist = disparityMap(grayL, grayR, pointLeft, pointRight, z);
			#if DEBUG 
				cout << "DIST IS : " << dist << endl; 
			#endif
			
			stream << fixed << setprecision(2) << dist;
			
			s = stream.str();
			dist_str = "Distance: " + s + " m";
		
			putText(original_image_L, dist_str, pointLeft, FONT_HERSHEY_PLAIN,1, Scalar(255,255,255),1,CV_AA);
			putText(original_image_R, dist_str, pointRight, FONT_HERSHEY_PLAIN,1, Scalar(255,255,255),1,CV_AA);
		}
		
		//-----------------------------------------------------------------
		
		CheckAndDraw(original_image_L, filteredDetections_L, filteredWeights_L);
		CheckAndDraw(original_image_R, filteredDetections_R, filteredWeights_R);
		
		
		imshow("L Vehicle Detection (Sequence)", original_image_L);
		imshow("R Vehicle Detection 2 (Sequence)", original_image_R);
		
		// Resets the points vector
		points.clear();
				
#if LOOP
		// Loop back the video
		if(i == fileCount - 1)
		{
			i = 0;
		}
#endif	
		
#if PRESS_NEXT
		if (waitKey(999999) == 37)
		{
			continue;
		}
#endif
		
		if(waitKey(30) == 27) //escape
		{ 
			return 1;
		}
			
	}
	
	time(&end);
	double seconds = difftime(end, start);
	cout << "Time taken: " << seconds << " seconds" << endl;
	//stop_counters();	
}

void PreProcessing(Mat &imageL, Mat &imageR)
{
	
	cvtColor(imageL, imageL, CV_BGR2GRAY);
	grayL = imageL;
	cvtColor(imageR, imageR, CV_BGR2GRAY);
	grayR = imageR;
	
	
	GaussianBlur(imageL, imageL, cv::GAUSSIAN_KERNEL_SIZE, 0, 0, BORDER_DEFAULT);
	GaussianBlur(imageR, imageR, cv::GAUSSIAN_KERNEL_SIZE, 0, 0, BORDER_DEFAULT);
}

void FileNameDetermine(int order, String &fileName_L, String &fileName_R)
{
	String FILE_PREFIX_L = "0000000";
	String FILE_PREFIX_R = "0000000";
	

	if (order < 10)
	{		
		fileName_L = FILE_PREFIX_L + "00" + to_string(order) + ".png";
		fileName_R = FILE_PREFIX_R + "00" + to_string(order) + ".png";
	}
	else if ((order >= 10) && (order < 100)) 
	{	
		fileName_L = FILE_PREFIX_L + "0" + to_string(order) + ".png";
		fileName_R = FILE_PREFIX_R + "0" + to_string(order) + ".png";			
	} 
	else 
	{		
		fileName_L = FILE_PREFIX_L + to_string(order) + ".png";	
		fileName_R = FILE_PREFIX_R + to_string(order) + ".png";			
	}
}


int FileCounter()
{
	DIR *dir;
	struct dirent *ent;
	
	int fileCount;
	
	if ((dir = opendir ("/home/pi/Desktop/CS700_RP/stereo_dataset/resize_left")) != NULL) {
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
	
	return fileCount;
	
}

void SetupHOG(HOGDescriptor &hog, Ptr<SVM> svm)
{
	hog.winSize = HOG_IMAGE_SIZE;
	hog.blockSize = HOG_BLOCK_SIZE;
	hog.cellSize = HOG_CELL_SIZE;
	hog.setSVMDetector(get_svm_detector(svm));
}

Ptr<SVM> LoadTrainingFile()
{
	clog << "Loading the Training file.. Please wait.. " << endl;
	Ptr<SVM> svm = StatModel::load<SVM>(YML_LOCATION);
	return svm;
}


void PointMatcher(Mat imageL, Mat imageR, vector<Rect>&detections_L, vector<Rect>&detections_R)
{
	
	try{
		//the larger  = the few/important keypoints
		//the smaller = the more/less-important keypoints
		int minHessian = 550; //<- Adjustable
		
		
		//left image
		for(int i = 0; i < detections_L.size(); i++)
		{
			Mat vehicleAreaLeft = imageL(detections_L[i]);
			
			
			//right image
			for(int j = 0; j < detections_R.size(); j++)
			{
				
				Mat vehicleAreaRight = imageR(detections_R[j]);
					
				//Inst. detector
				Ptr<SIFT> featureDetector = SIFT::create();
				//featureDetector->setHessianThreshold(minHessian);
					
				//Declare vars
				vector<KeyPoint> keypointLeftImage, keypointRightImage;
				Mat descriptorLeftImage, descriptorRightImage;
					
				featureDetector->detectAndCompute(vehicleAreaLeft, Mat(), keypointLeftImage, descriptorLeftImage);
				featureDetector->detectAndCompute(vehicleAreaRight, Mat(), keypointRightImage, descriptorRightImage);
					
				//Declare FLANN matcher
				FlannBasedMatcher matcher;
				vector<DMatch> matches; //save results to 'matches'
				
				if(descriptorLeftImage.empty() || descriptorRightImage.empty())
				{
					continue;
				}
				
				matcher.match(descriptorLeftImage, descriptorRightImage, matches);

				double max_dist = 0;
				double min_dist = 100;
				double dist = 0;
				
				
				for(int i = 0; i < descriptorLeftImage.rows; i++)
				{
					dist = matches[i].distance;
						
					if(dist < min_dist)
					{
						min_dist = dist;
					}
					if(dist > max_dist)
					{
						max_dist = dist;
					}
				}
			
				
				
				vector<DMatch> good_matches;
				for(int i = 0; i < descriptorLeftImage.rows; i++)
				{
					if(matches[i].distance <= max(2 * min_dist, 0.02))
					{
						good_matches.push_back(matches[i]);
					}
				}

				
				if(good_matches.size() >= 5)
				{
					
						tempPoints.leftPoint = (Point(detections_L[i].x, detections_L[i].y + 105));
						tempPoints.rightPoint = (Point(detections_R[j].x, detections_R[j].y + 105));
						
						#if DEBUG 
						cout << "LEFT POINTS: " << tempPoints.leftPoint << endl;
						cout << "RIGHT POINTS: " << tempPoints.rightPoint << endl;
						#endif

						points.push_back(tempPoints);
						
						#if DEBUG 
						imshow("SURF Matched L", vehicleAreaLeft);
						imshow("SURF Matched R", vehicleAreaRight);
						#endif
				}
			}	
		}
	}
	catch(int e)
	{
		cout << "An exception occurred. Exception Nr. " << e << '\n';
	}
}

float disparityMap(Mat imageL, Mat imageR, Point pointLeft, Point pointRight, int index)
{

	if (((pointLeft.x + 105) > imageL.size().width) || ((pointLeft.y) > imageL.size().height))
	{
		#if DEBUG 
			cout << "LEFT TRUE" <<endl;
		#endif
		return 0;
	}
	
	if (((pointRight.x + 105) > imageR.size().width) || ((pointRight.y) > imageR.size().height))
	{
		#if DEBUG
			cout << "RIGHT TRUE" <<endl;
		#endif
		return 0;
	}
	
	
	const double BASELINE = -(-3.745166) / 6.471884; // Distance between the two cameras
	const double FOCAL = 647.1884; // Focal Length in pixels
	float final_dist = 0;
	
	Ptr<StereoBM> sbm_crop = StereoBM::create(16, 17);
	
	Mat disparity, disp8;	
	Mat ROI_disp_L, ROI_disp_R;	

	Rect roi_L, roi_R;
	
	
	roi_L = Rect(pointLeft.x, pointLeft.y - 105,105,105);
	ROI_disp_L = Mat(imageL, roi_L);
	
	roi_R = Rect(pointRight.x, pointLeft.y - 105,105,105);
	ROI_disp_R = Mat(imageR, roi_R);
	
	
	String leftImg = "LEFT IMAGE"; //_" + to_string(index);
	String rightImg = "RIGHT IMAGE";//_" + to_string(index);
	
	#if DEBUG 
		imshow(leftImg, ROI_disp_L);
		imshow(rightImg, ROI_disp_R);
	#endif
	
	disparity = Mat(ROI_disp_L.size().height, ROI_disp_L.size().width, CV_16S);
	
	sbm_crop->setUniquenessRatio(15);
	sbm_crop->setTextureThreshold(0.0002);
	sbm_crop->compute(ROI_disp_L, ROI_disp_R, disparity);
	
	normalize(disparity, disp8, 0, 255, CV_MINMAX, CV_8U);
	
	String dispString = "DISPARITY"; //_" + to_string(index);
	
	#if DEBUG 
		imshow(dispString, disp8);
	#endif
	#if DEBUG 
		cout << "pixel value: " << (int)disp8.at<unsigned char>(60, 60) << endl;
	#endif
	final_dist = (FOCAL*BASELINE) / (int)disp8.at<unsigned char>(60, 60);
	#if DEBUG 
		cout << "distance : " << final_dist << " m" << endl;
	#endif
	
	
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

//reduce the number of detected rect by filtering out lower weights rect
void HOGConfidenceFilter(vector<Rect> &detections, vector<double> &foundWeights, vector<Rect> &new_detections, vector<double> &new_foundWeights)
{
	#if DEBUG 
	cout << "Detection count before conf filter: " << detections.size() << endl;
	#endif 
	double confidence;

	for(size_t i = 0; i < detections.size(); i++)
	{
		confidence = foundWeights[i] * foundWeights[i];
		if((confidence > CONFIDENCE_THRESHOLD) || BYPASS_CONFIDENCE_CHECK)
		{			
			new_detections.push_back(detections[i]);
			new_foundWeights.push_back(foundWeights[i]);
		} 
	}
	#if DEBUG
		cout << "Detection count after conf filter: " << new_detections.size() << endl;
	#endif
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





// Performance Counters
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags)
{
    int ret;

    ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                   group_fd, flags);
    return ret;
}


static void setup_counters(void)
{
	
    struct perf_event_attr pe;

    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    inst_fd = perf_event_open(&pe, 0, -1, -1, 0);
    if (inst_fd == -1) {
       fprintf(stderr, "Error opening leader %llx\n", pe.config);
       exit(EXIT_FAILURE);
    }
    
    ioctl(inst_fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(inst_fd, PERF_EVENT_IOC_ENABLE, 0);
	
}

static void stop_counters(void)
{
	
	ioctl(inst_fd, PERF_EVENT_IOC_DISABLE, 0);
    read(inst_fd, &inst_count, sizeof(long long));

    printf("Used %lld cycles\n", inst_count);

    close(inst_fd);	
    
}
	
	

