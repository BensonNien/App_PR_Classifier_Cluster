#include "BayesianDecision.h"

void BayesianClassifer(Mat& XsubMean, Mat& Covar_train, double prior, double& posterior, int XsubMeanRows, int XsubMeanCols, int Covar_trainRows, int Covar_trainCols)
{
	Mat XsubMean_tp = Mat::zeros(XsubMeanRows, XsubMeanCols, CV_32FC1);
	Mat Covar_train_inv = Mat::zeros(Covar_trainRows, Covar_trainCols, CV_32FC1);
	Mat term1 = Mat::zeros(1, 1, CV_32FC1);
	Mat term2 = Mat::zeros(1, 1, CV_32FC1);
	Mat term3 = Mat::zeros(1, 1, CV_32FC1);

	double det_Covar_train_val = 0.0;
	double term1_val = 0.0;
	double term2_val = 0.0;
	double term3_val = 0.0;


	transpose(XsubMean, XsubMean_tp);
	invert(Covar_train, Covar_train_inv, DECOMP_LU);
	term1 = XsubMean_tp * Covar_train_inv * XsubMean;
	term1_val = 0.5 * term1.at<float>(0, 0);
	det_Covar_train_val = determinant(Covar_train);
	term2_val = 0.5 * log(det_Covar_train_val);
	term3_val = log(prior);
	posterior = term1_val + term2_val - term3_val;
}

void MinPosterior(double lposterior1, double lposterior2, double lposterior3, double targetposterior, int& lerror)
{
	double min1 = min(lposterior1, lposterior2);
	double min2 = min(min1, lposterior3);
	if (min2 != targetposterior)
	{
		lerror = lerror + 1;
	}
}
