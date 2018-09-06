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
	
	Mat resize_imageL, resize_imageR;
	
	for (int i = IMG_STARTING_SEQUENCE; i < IMG_STOPPING_SEQUENCE + 10; i++)
	{
		cout << i << endl;
		FileNameDetermine(i, fileName_L, fileName_R);
		
		image_L = imread(L_CAMERA_SRC_DIR + fileName_L);
		image_R = imread(R_CAMERA_SRC_DIR + fileName_R);
		
		original_image_L = image_L;
		original_image_R = image_R;
		
		PreProcessing(image_L, image_R);
		
		//resize image for HOG processing
		resize(image_L, resize_imageL, Size(), 0.5, 0.5);
		//resize(image_R, resize_imageR, Size(), 0.5, 0.5);
		
		//cout << "Size original: " << image_L.size() << endl;
		//cout << "Size modified: "  << resize_imageL.size() << endl;
		
		//crop the image by ROI
		//Rect roi = Rect(0,0, ROI_X, ROI_Y);
		Rect roi = Rect(0, 0, image_L.size().width, image_L.size().height);
		
		image_L = Mat(image_L, roi);
		image_R = Mat(image_R, roi);
		
		//change image_L to resize_imageL if needed
		#if SCALED
			hog.detectMultiScale(resize_imageL, detections_L, weights_L);
		#else
			hog.detectMultiScale(image_L, detections_L, weights_L);
		#endif
		filteredDetections_L.clear();
		filteredWeights_L.clear();
		
		
		HOGConfidenceFilter(detections_L, weights_L, filteredDetections_L, filteredWeights_L);		
		depthMap = DepthMap(image_L, image_R);
		imshow("Depth Map", depthMap);
			
		//-----------------------------------------------------------------
		
		
		CheckAndDraw(original_image_L, filteredDetections_L, filteredWeights_L);		
		imshow("L Vehicle Detection (Sequence)", original_image_L);
		
					
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
	
	#if USE_GAUSSIAN
	GaussianBlur(imageL, imageL, cv::GAUSSIAN_KERNEL_SIZE, 0, 0, BORDER_DEFAULT);
	GaussianBlur(imageR, imageR, cv::GAUSSIAN_KERNEL_SIZE, 0, 0, BORDER_DEFAULT);
	#endif
}


void FileNameDetermine(int order, String &fileName_L, String &fileName_R)
{
	//File Prefix
	String FILE_PREFIX_L = "I1_000";
	String FILE_PREFIX_R = "I2_000";
	

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
	
	String dist_str;
	
	Point points;
	
	for(size_t i = 0; i < detections.size(); i++)
	{
		
		confidence = foundWeights[i] * foundWeights[i];
		confidence_colour = Scalar(0, confidence * 200, 0);
		
		#if SCALED 
			detections[i].height *= 2;
			detections[i].width *= 2;
			detections[i].x *= 2;
			detections[i].y *=2;
		#endif 
		
		rectangle(image, detections[i], confidence_colour, image.cols / 400 + 1);
		
		cout << detections[i] << endl;
		
		points = Point(detections[i].x, detections[i].y); 
		
		dist_str = "Distance: " + estimateDistance(detections[i]) + " m";
		putText(image, dist_str, points, FONT_HERSHEY_PLAIN,1, Scalar(255,255,255),1,CV_AA);
		
		 
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

	max_disp/=2;
    if(max_disp%16!=0)
    {
		max_disp += 16-(max_disp%16);
    }
        
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
    // Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,wsize);
    // left_matcher->setMode(StereoSGBM::MODE_SGBM);
    
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
    double sigma = 1.5;       // hardcode
	int vis_mult = 1;

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

string estimateDistance(Rect &detections) 
{
	
	const double BASELINE = -(-3.745166) / 6.471884; // Distance between the two cameras
	const double FOCAL = 647.1884; // Focal Length in pixels
		
	int x = detections.x + (detections.size().width / 2);
	
	//add a quarter to prevent distance estimation on the windscreen
	int y = detections.y + (detections.size().height / 2) + (detections.size().height / 4);
	
	int pixelValue = (int)depthMap.at<unsigned char>(y,x);
	
	if (pixelValue == 0) return "-";
	
	double distance = (FOCAL * BASELINE) / pixelValue;
	
	std::ostringstream strs;
	
	strs << fixed << setprecision(2) << distance;
	std::string dist = strs.str();
		
	return dist;
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
	
	

