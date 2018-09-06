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
#define YML_LOCATION "/home/pi/Desktop/CS700_RP/vehicle_detector.yml"
#define L_CAMERA_SRC_DIR "/home/pi/Desktop/CS700_RP/stereo_dataset/left/"
#define R_CAMERA_SRC_DIR "/home/pi/Desktop/CS700_RP/stereo_dataset/right/"

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
#define USE_GAUSSIAN 0
#define GAUSSIAN_KERNEL_SIZE Size(3,3)


#define SCALED 1

//Starting image sequence
#define IMG_STARTING_SEQUENCE 323
#define IMG_STOPPING_SEQUENCE 400

//Settings
#define CONFIDENCE_THRESHOLD 0.5
#define BYPASS_CONFIDENCE_CHECK 0

//ROI, cropping x and y
#define ROI_X 
#define ROI_Y

//Looping and Debugging
#define LOOP_IMAGES 0
#define PRESS_NEXT 0


//variables
long long inst_count;
int inst_fd;

//functions declarations
vector<float> get_svm_detector(const Ptr<SVM> &svm);
vector<Rect> HOGConfidenceFilter(vector<Rect> &detections, vector<double> &foundWeights);

Ptr<SVM> LoadTrainingFile();
int FileCounter();
void SetupHOG(HOGDescriptor &hog, Ptr<SVM> svm);

void PreProcessing(Mat &imageL, Mat &imageR);
void FileNameDetermine(int order, String &fileName_L, String &fileName_R);

void CheckAndDraw(Mat &image, vector<Rect> &detections, vector<double> &foundWeights);
void HOGConfidenceFilter(vector<Rect> &detections, vector<double> &foundWeights, vector<Rect> &new_detections, vector<double> &new_foundWeights);

static void setup_counters(void);
static void stop_counters(void);
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, 
int cpu, int group_fd, unsigned long flags);

Mat grayL, grayR;

Mat depthMap;
cv::Mat DepthMap(cv::Mat &imageL, cv::Mat &imageR);
string estimateDistance(Rect &detections);

int main()
{
	
	Ptr<SVM> svm;
	svm = LoadTrainingFile();
	if(!svm)
	{
		return 0;
	}
	
	//Prepare HOG Descriptor to detect vehicles
	HOGDescriptor hog;
	SetupHOG(hog, svm);
	
	
	Mat image_L;
	String fileName_L = "I1_000184.png";
	
	image_L = imread(L_CAMERA_SRC_DIR + fileName_L);
	
	cvtColor(image_L, image_L, CV_BGR2GRAY);
	
	vector<Rect> detections_L, detections_R;
	vector<double> weights_L, weights_R;
	
	time_t start, end;
	
	cout << "Press Enter to continue" << endl;
	cin.get();	
	
	time(&start);
	hog.detectMultiScale(image_L, detections_L, weights_L);
	time(&end);
	
	double seconds = difftime(end, start);
	
	cout << "Time taken: " << seconds << " seconds" << endl;
	
	if(detections_L.size() != 0)
	{
		cout << "good detection" << endl;
	}			
	
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




	
	

