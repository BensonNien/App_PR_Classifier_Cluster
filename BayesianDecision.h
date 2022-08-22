#pragma once
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp" 


using namespace std;
using namespace cv;

void BayesianClassifer(Mat& XsubMean, Mat& Covar_train, double prior, double& posterior, int XsubMeanRows, int XsubMeanCols, int Covar_trainRows, int Covar_trainCols);
void MinPosterior(double lposterior1, double lposterior2, double lposterior3, double targetposterior, int& lerror);