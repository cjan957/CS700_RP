#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

//SIFT and KEYPOINT STUFF
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
//end SIFT and KEYPOINT STUFF

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

// -- Timing --

#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <asm/unistd.h>



// -------------------------

using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace cv::xfeatures2d;

//#define YML_LOCATION "vehicle_detector_filter.yml"
#define YML_LOCATION "/home/pi/Desktop/CS700_RP/Gammar0.5/vehicle_detector_filter.yml"

#define IMAGE_SIZE Size(64,64)

#define SEQUENCE_SET 1 // 1 for sequence of images, 0 for a single shot (include L and R), specify below
#define CAMERA_1_LOCATION "/home/pi/Desktop/CS700_RP/stereo_dataset/resize_left/I1_000163.png"
#define CAMERA_2_LOCATION "/home/pi/Desktop/CS700_RP/stereo_dataset/resize_right/I2_000163.png"
#define START_AT_SEQUENCE 100 // <- specify STARTING SEQUENCE HERE
#define CONFIDENCE_THRESHOLD 0.5 //specify confidence threshold for when bypass is 0
#define BYPASS_CONFIDENCE_CHECK 0 //draw black boxes when not confidence

#define LOOP 0

vector<float> get_svm_detector(const Ptr<SVM> &svm);
void CheckAndDraw(Mat &image, vector<Rect> &detections, vector<double> &foundWeights);
Point getPoints(Mat &image, vector<Rect> &detections, vector<double> &foundWeights);

float disparityMap(Mat imageL, Mat imageR, vector<Rect> &detections_L, vector<double> &foundWeights_L, vector<Rect> &detections_R, vector<double> &foundWeights_R);

void SURFMatcher(Mat imageL, Mat imageR, vector<Rect>&detections_L, vector<Rect>&detections_R);

vector<Rect> HOGConfidenceFilter(vector<Rect> &detections, vector<double> &foundWeights);

static void setup_counters(void);
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, 
	int cpu, int group_fd, unsigned long flags);

