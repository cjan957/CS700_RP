#include "detect_vehicle.hpp"
#include "CorrespondingPoint.hpp"

using namespace cv;


Point leftPoint;
Point rightPoint;
	
void setLeftPoint(Point l)
{
	leftPoint = l;	
}

void setRightPoint(Point r)
{
	rightPoint = r;	
}				

Point getLeftPoint()
{
	return leftPoint;	
}		

Point getRightPoint()
{
	return rightPoint;	
}		

