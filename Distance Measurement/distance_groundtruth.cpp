// Brief Sample of using OpenCV dnn module in real time with device capture, video and image.
// VIDEO DEMO: https://www.youtube.com/watch?v=NHtRlndE2cg

#include "stdafx.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <algorithm>
#include <cstdlib>

#include <iomanip>
#include <sstream>
#include <stdlib.h>

using namespace std;
using namespace cv;
using namespace cv::dnn;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image
vector<string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

// Displays the distance based on where it was clicked
void distanceEstimator(int event, int x, int y, int flags, void* userdata);

Ptr<StereoBM> sbm;

Mat blob, g1, g2, disparity, disp8, true_map;

const double BASELINE = 0.54; //Karlsruhe:-(-3.745166) / 6.471884; // Distance between the two cameras
const double FOCAL = 721; // 645.24; //Karlsruhe : 647.1884; // Focal Length in pixels


Mat projectionMatrix0;

const string imageNumber = "000071";

void getProjectionMatrix() {

	string textline;
	int text;
	string buf;
	vector<string> tokens;
	vector<double> tempMatrix;

	int i = 0;

	ifstream myfile("C:/Users/Josh/Documents/Uni/Part4-Project/data_object_calib/training/calib/" + imageNumber + ".txt");

	if (myfile.is_open())
	{
		while (getline(myfile, textline))
		{

			stringstream ss(textline);

			while (ss >> buf) {

				tokens.push_back(buf);
				tempMatrix.push_back(atof(buf.c_str()));
			}

			if (tokens[0] != "P2:") {
				tokens.clear();
				tempMatrix.clear();
			}
			else {
				// cout << textline << '\n';
				break;
			}


			// break;
		}
		myfile.close();
	}
	else
	{
		cout << "Unable to open file " << endl;
		return;

	}

	projectionMatrix0 = (cv::Mat_<double>(3, 4) <<
		tempMatrix[1], tempMatrix[2], tempMatrix[3], tempMatrix[4],
		tempMatrix[5], tempMatrix[6], tempMatrix[7], tempMatrix[8],
		tempMatrix[9], tempMatrix[10], tempMatrix[11], tempMatrix[12]);

}

void getAllDistances() {

	string textline;
	int text;
	string buf;
	vector<string> tokens;
	Mat result;

	ifstream myfile("C:/Users/Josh/Documents/Uni/Part4-Project/Dataset/label_2/" + imageNumber + ".txt");

	if (myfile.is_open())
	{
		while (getline(myfile, textline))
		{

			stringstream ss(textline);

			while (ss >> buf) {
				tokens.push_back(buf);
			}

			if (tokens[0] != "DontCare") {

				Mat coordinates = (cv::Mat_<double>(4, 1) << atof(tokens[10].c_str()), atof(tokens[11].c_str()), atof(tokens[12].c_str()), 1);
				cout << "Coordinates " << coordinates << endl;
				result = projectionMatrix0 * coordinates;
				cout << tokens[0] << " distance: " << result.at<double>(2, 0) << " m" << endl;
				// cout << textline << '\n';
			}

			tokens.clear();

		}
		myfile.close();
	}
	else
	{
		cout << "Unable to open file " << endl;
		return;

	}

}

int main()
{

	getProjectionMatrix();

	// Get coordinates
	getAllDistances();

	// -----------------------------------------------------------------------

	// Load names of classes
	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// Give the configuration and weight files for the model
	String modelConfiguration = "yolov3.cfg";
	String modelWeights = "yolov3.weights";

	// Load the network
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Create a window
	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL); // Makes the window size bigger

	Mat leftImg = imread("C:/Users/Josh/Documents/Uni/Part4-Project/Dataset/LEFT/" + imageNumber + ".png");
	Mat rightImg = imread("C:/Users/Josh/Documents/Uni/Part4-Project/Dataset/RIGHT/" + imageNumber + ".png");

	imshow("Original Image", leftImg);

	cvtColor(leftImg, g1, COLOR_BGR2GRAY);
	cvtColor(rightImg, g2, COLOR_BGR2GRAY);

	// Remove noise by blurring with a Gaussian filter
	GaussianBlur(g1, g1, Size(3, 3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(g2, g2, Size(3, 3), 0, 0, BORDER_DEFAULT);

	Size imagesize = g1.size();

	disparity = Mat(imagesize.height, imagesize.width, CV_16S);

	int iValueForNumDisp = 128;
	int iValueForBlockSize = 37; // 39

	sbm = StereoBM::create(iValueForNumDisp, iValueForBlockSize);

	sbm->compute(g1, g2, disparity);
	disparity.convertTo(true_map, CV_32F, 1.0 / 16.0, 0.0);

	int x = 488;
	int y = 218;

	// double depth = 0.54 * 721 / (true_map.at<float>(y, x));
	// double depth = (0.54 * 721) / (int)disp8.at<unsigned char>(y, x);

	// cout << "Pixel Value : " << true_map.at<float>(y, x) << endl;
	// cout << "DEPTH IS: " << depth << " m" << endl;

	normalize(disparity, disp8, 0, 255, CV_MINMAX, CV_8U);
	imshow("disp", disp8);

	//set the callback function for any mouse event
	setMouseCallback("disp", distanceEstimator, NULL);

	if (!leftImg.data)
	{
		return -1;
	}

	// Create a 4D blob from a frame.
	blobFromImage(leftImg, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

	//Sets the input to the network
	net.setInput(blob);

	// Runs the forward pass to get output of the output layers
	vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));

	// Remove the bounding boxes with low confidence
	postprocess(leftImg, outs);

	// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("Inference time for a frame : %.2f ms", t);
	putText(leftImg, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

	// Write the frame with the detection boxes
	Mat detectedFrame;
	leftImg.convertTo(detectedFrame, CV_8U);

	imshow(kWinName, leftImg);

	waitKey(0);

	return 0;

} // main


void distanceEstimator(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "pixel value: " << (int)disp8.at<unsigned char>(y, x) << endl;
		cout << "distance : " << (FOCAL*BASELINE) / (int)disp8.at<unsigned char>(y, x) << " m" << endl;
	}
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

string estimateDistance(int y, int x, Mat roi) {

	if (!roi.data) return "-";

	int pixelValue = (int)roi.at<unsigned char>(y, x);

	if (pixelValue == 0) return "-";

	double distance = (FOCAL*BASELINE) / pixelValue;

	cout << "Distance is : " << distance << " m" << endl;

	std::ostringstream strs;

	strs << fixed << setprecision(2) << distance;
	std::string dist = strs.str();

	return dist;
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	if ((!classes[classId].compare("car")) | (!classes[classId].compare("truck "))) {
		cout << classes[classId] << endl;
		return;
	}

	if (left < 0) {
		left = 0;
	}

	if (right > frame.size().width) {
		right = frame.size().width;
	}

	if (top < 0) {
		top = 0;
	}

	if (bottom > frame.size().height) {
		bottom = frame.size().height;
	}

	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	// cout << "Image locations: " << left << "," << top << " Size: " << (right - left) << "," << (bottom - top) << endl;

	Mat roi;

	try
	{
		Rect cropped = Rect(left, top, (right - left), (bottom - top));
		roi = Mat(disp8, cropped);
		imshow("roi", roi);
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
	}



	int y, x;

	y = (right - left) / 2;
	x = (bottom - top) / 2;

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label + " Distance: " + estimateDistance(y, x, roi);
	}
	
	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}