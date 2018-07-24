#include "detect_vehicle.hpp"

int main()
{	 
	// Uncomment to make it run on a single core
	setNumThreads(0);
	
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
	Ptr<StereoBM> sbm = StereoBM::create(64, 31);
	
	Mat ROI_disp_L, ROI_disp_R;	
	
	vector<Rect> detections;
	vector<double> foundWeights;
	
	vector<Rect> detections_2;
	vector<double> foundWeights_2;
	

	//after HOGConfidenceFilter
	vector<Rect> filteredDetections_L; 
	vector<Rect> filteredDetections_R;
	
	vector<double> filteredFoundWeights_L;
	vector<double> filteredFoundWeights_R;

	float dist;
	Point pointLeft;
	Point pointRight;
	String dist_str;
	

#if SEQUENCE_SET	//sequence of images (video)
		
	int fileCount;
	
	DIR *dir;
	struct dirent *ent;
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
	
	cout << "LENGTH IS " << fileCount<< endl;
	
	int i = 0;
	
	string openFile = "I1_000";
	string openSecondFile = "I2_000";
	
	string testFile = "";
	string testFile_2 = "";
	
	string direct= "/home/pi/Desktop/CS700_RP/stereo_dataset/resize_left/";
	string direct_right = "/home/pi/Desktop/CS700_RP/stereo_dataset/resize_right/";

	cout << "TIMER STARTED " << endl;
	
	setup_counters();
	
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
		Rect roi = Rect(0, 0, blurred.size().width, blurred.size().height);
		//Rect roi = Rect(0, 0, 392, blurred.size().height);
		ROI_L = Mat(blurred, roi);
		ROI_R = Mat(blurred_2, roi);
		
		
		hog.detectMultiScale(ROI_L, detections, foundWeights);
		hog.detectMultiScale(ROI_R, detections_2, foundWeights_2);
		
		filteredDetections_L.clear();
		filteredDetections_R.clear();
		filteredFoundWeights_L.clear();
		filteredFoundWeights_R.clear();
		
		// HOGConfidenceFilter
		HOGConfidenceFilter(detections, foundWeights, filteredDetections_L, filteredFoundWeights_L);
		HOGConfidenceFilter(detections_2, foundWeights_2, filteredDetections_R, filteredFoundWeights_R);
		//NOTE : foundWeights are no longer valid from this line
		
		SURFMatcher(ROI_L, ROI_R, filteredDetections_L, filteredDetections_R);
		
		//--------------------DISPARITY---------------------------
		disparity = Mat(grayL.size().height, grayL.size().width, CV_16S);
		
		sbm->setUniquenessRatio(15);
		sbm->setTextureThreshold(0.0002);
		sbm->compute(grayL, grayR, disparity);
		
		normalize(disparity, disp8, 0, 255, CV_MINMAX, CV_8U);
		
		imshow("Disparity Map - FULL", disp8);
		
		stringstream stream;
		string s;
		
		
		for (int z = 0; z < points.size() ; z++)
		{

			pointLeft = points.at(z).leftPoint;
			pointRight = points.at(z).rightPoint;
				
			dist = disparityMap(grayL, grayR, pointLeft, pointRight, z);
			cout << "DIST IS : " << dist << endl;
			
			stream << fixed << setprecision(2) << dist;
			
			s = stream.str();
			dist_str = "Distance: " + s + " m";
		
			putText(image, dist_str, pointLeft, FONT_HERSHEY_PLAIN,1, Scalar(255,255,255),1,CV_AA);
			putText(image_2, dist_str, pointRight, FONT_HERSHEY_PLAIN,1, Scalar(255,255,255),1,CV_AA);
		}
		
		//-----------------------------------------------------------------
		
		CheckAndDraw(image, filteredDetections_L, filteredFoundWeights_L);
		CheckAndDraw(image_2, filteredDetections_R, filteredFoundWeights_R);
		
		imshow("L Vehicle Detection (Sequence)", image);
		imshow("R Vehicle Detection 2 (Sequence)", image_2);
		
		cout << "Image : " << i << endl;
	
		// Resets the points vector
		points.clear();
	
		
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
	
	ioctl(inst_fd, PERF_EVENT_IOC_DISABLE, 0);
    read(inst_fd, &inst_count, sizeof(long long));

    printf("Used %lld cycles\n", inst_count);

    close(inst_fd);	
	
	
	
#else //STATIC IMAGES
	
	const double BASELINE = -(-3.745166) / 6.471884; // Distance between the two cameras
	const double FOCAL = 647.1884; // Focal Length in pixels
	
	image = imread(CAMERA_1_LOCATION);
	image_2 = imread(CAMERA_2_LOCATION);
		
	// ------------ DISPARITY -------------
	// Grayscale the image
	cvtColor(image, grayL, CV_BGR2GRAY);
	cvtColor(image_2, grayR, CV_BGR2GRAY);
	
	// Remove the noise
	GaussianBlur(grayL, blurred, Size(3,3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(grayR, blurred_2, Size(3,3), 0, 0, BORDER_DEFAULT);
	
	disparity = Mat(blurred.size().height, blurred.size().width, CV_16S);
	
	sbm->setUniquenessRatio(15);
	sbm->setTextureThreshold(0.0002);
	sbm->compute(grayL, grayR, disparity);
	
	normalize(disparity, disp8, 0, 255, CV_MINMAX, CV_8U);
	
	imshow("Disparity Map", disp8);
	
	cout << "pixel value (original): " << (int)disp8.at<unsigned char>(110,327) << endl;
	cout << "distance : " << (FOCAL*BASELINE) / (int)disp8.at<unsigned char>(110,327) << " m" << endl;
	
	//-----------------------------------------------------//

	hog.detectMultiScale(blurred, detections, foundWeights);
	hog.detectMultiScale(blurred_2, detections_2, foundWeights_2);
	
	//add SURF stuff here?
	
	//-------------------------------------------------------//
			
	dist = disparityMap(blurred, blurred_2, detections, foundWeights, detections_2, foundWeights_2);
		
	stringstream stream;
	stream << fixed << setprecision(2) << dist;
	string s = stream.str();
	dist_str = "Distance: " + s + " m";
	
	pointLeft = getPoints(blurred, detections, foundWeights);
	pointRight = getPoints(blurred_2, detections_2, foundWeights_2);
	
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

void SURFMatcher(Mat imageL, Mat imageR, vector<Rect>&detections_L, vector<Rect>&detections_R)
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
						
						cout << "LEFT POINTS: " << tempPoints.leftPoint << endl;
						cout << "RIGHT POINTS: " << tempPoints.rightPoint << endl;

						points.push_back(tempPoints);
						
						
						imshow("SURF Matched L", vehicleAreaLeft);
						imshow("SURF Matched R", vehicleAreaRight);
					
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
		cout << "LEFT TRUE" <<endl;
		return 0;
	}
	
	if (((pointRight.x + 105) > imageR.size().width) || ((pointRight.y) > imageR.size().height))
	{
		cout << "RIGHT TRUE" <<endl;
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
	
	imshow(leftImg, ROI_disp_L);
	imshow(rightImg, ROI_disp_R);
	
	disparity = Mat(ROI_disp_L.size().height, ROI_disp_L.size().width, CV_16S);
	
	sbm_crop->setUniquenessRatio(15);
	sbm_crop->setTextureThreshold(0.0002);
	sbm_crop->compute(ROI_disp_L, ROI_disp_R, disparity);
	
	normalize(disparity, disp8, 0, 255, CV_MINMAX, CV_8U);
	
	String dispString = "DISPARITY"; //_" + to_string(index);
	
	imshow(dispString, disp8);
	
	cout << "pixel value: " << (int)disp8.at<unsigned char>(60, 60) << endl;
	final_dist = (FOCAL*BASELINE) / (int)disp8.at<unsigned char>(60, 60);
	cout << "distance : " << final_dist << " m" << endl;
	
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
	cout << "Detection count before conf filter: " << detections.size() << endl;
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
	cout << "Detection count after conf filter: " << new_detections.size() << endl;
	
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
	
	
	
	
