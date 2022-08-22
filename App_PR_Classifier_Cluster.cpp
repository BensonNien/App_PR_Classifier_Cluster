// App_PR_Classifier_Cluster.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>

#include <stdio.h>
#include <math.h> //log
//#include <Windows.h>
//#include <tchar.h>
//#include <libloaderapi.h>
//#include <wchar.h>

#include "BayesianDecision.h"

using namespace std;

int main()
{
	//HMODULE hDll = LoadLibrary(_T("C:\\Lib_3rd_Party\\opencv_3416_vc14_vc15\\build\\x64\\vc14\\bin"));
	//if (!hDll || hDll == INVALID_HANDLE_VALUE) {
	//	_tprintf(_T("unable to load libraray"));
	//	return 1;
	//}
	
	
	/***read txt file***/
	char _filename_1_train[64] = "./input_data/wine_class1_train.txt";
	char _filename_2_train[64] = "./input_data/wine_class2_train.txt";
	char _filename_3_train[64] = "./input_data/wine_class3_train.txt";
	char _filename_1_test[64] = "./input_data/wine_class1_test.txt";
	char _filename_2_test[64] = "./input_data/wine_class2_test.txt";
	char _filename_3_test[64] = "./input_data/wine_class3_test.txt";

	char _resultofclassification[64] = "./output/result_of_classification.txt";

	ofstream result_of_classification(_resultofclassification, ios::out);

	/****read 3 txt file****/

	ifstream in_file_1_train(_filename_1_train);
	ifstream in_file_2_train(_filename_2_train);
	ifstream in_file_3_train(_filename_3_train);
	ifstream in_file_1_test(_filename_1_test);
	ifstream in_file_2_test(_filename_2_test);
	ifstream in_file_3_test(_filename_3_test);

	Mat matrix_a1_train = Mat(30, 13, CV_32FC1);
	Mat matrix_a2_train = Mat(36, 13, CV_32FC1);
	Mat matrix_a3_train = Mat(24, 13, CV_32FC1);
	Mat matrix_a1_test = Mat(13, 29, CV_32FC1);
	Mat matrix_a2_test = Mat(13, 35, CV_32FC1);
	Mat matrix_a3_test = Mat(13, 24, CV_32FC1);
	Mat Mean_a1_train = Mat::zeros(13, 1, CV_32FC1);
	Mat Mean_a2_train = Mat::zeros(13, 1, CV_32FC1);
	Mat Mean_a3_train = Mat::zeros(13, 1, CV_32FC1);
	Mat Covar_a1_1_train = Mat::zeros(13, 13, CV_32FC1);
	Mat Covar_a1_2_train = Mat::zeros(13, 13, CV_32FC1);
	Mat Covar_a2_1_train = Mat::zeros(13, 13, CV_32FC1);
	Mat Covar_a2_2_train = Mat::zeros(13, 13, CV_32FC1);
	Mat Covar_a3_1_train = Mat::zeros(13, 13, CV_32FC1);
	Mat Covar_a3_2_train = Mat::zeros(13, 13, CV_32FC1);

	Mat Mean_a1_train_tp1 = Mat(13, 1, CV_32FC1);
	Mat Mean_a2_train_tp1 = Mat(13, 1, CV_32FC1);
	Mat Mean_a3_train_tp1 = Mat(13, 1, CV_32FC1);

	double prior_class1 = 30.0 / 90.0;
	double prior_class2 = 36.0 / 90.0;
	double prior_class3 = 24.0 / 90.0;


	//printf("==================class 1 for training==================\n");
	for (int i = 0; i < 30; i++)
	{
		for (int j = 0; j < 13; j++)
		{
			in_file_1_train >> matrix_a1_train.at<float>(i, j);
		}
	}

	//printf("==================class 2 for training==================\n");
	for (int i = 0; i < 36; i++)
	{
		for (int j = 0; j < 13; j++)
		{
			in_file_2_train >> matrix_a2_train.at<float>(i, j);
		}
	}

	//printf("==================class 3 for training==================\n");
	for (int i = 0; i < 24; i++)
	{
		for (int j = 0; j < 13; j++)
		{
			in_file_3_train >> matrix_a3_train.at<float>(i, j);
		}
	}

	//printf("==================class 1 for testing==================\n");
	for (int i = 0; i < 29; i++)
	{
		for (int j = 0; j < 13; j++)
		{
			in_file_1_test >> matrix_a1_test.at<float>(j, i);
		}
	}

	//printf("==================class 2 for testing==================\n");
	for (int i = 0; i < 35; i++)
	{
		for (int j = 0; j < 13; j++)
		{
			in_file_2_test >> matrix_a2_test.at<float>(j, i);
		}
	}

	//printf("==================class 3 for testing==================\n");
	for (int i = 0; i < 24; i++)
	{
		for (int j = 0; j < 13; j++)
		{
			in_file_3_test >> matrix_a3_test.at<float>(j, i);
		}
	}


	calcCovarMatrix(matrix_a1_train, Covar_a1_1_train, Mean_a1_train, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);
	transpose(Mean_a1_train, Mean_a1_train_tp1);
	Covar_a1_1_train.convertTo(Covar_a1_2_train, CV_32FC1, 1 / 30.0);


	calcCovarMatrix(matrix_a2_train, Covar_a2_1_train, Mean_a2_train, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);
	transpose(Mean_a2_train, Mean_a2_train_tp1);
	Covar_a2_1_train.convertTo(Covar_a2_2_train, CV_32FC1, 1 / 36.0);


	calcCovarMatrix(matrix_a3_train, Covar_a3_1_train, Mean_a3_train, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);
	transpose(Mean_a3_train, Mean_a3_train_tp1);
	Covar_a3_1_train.convertTo(Covar_a3_2_train, CV_32FC1, 1 / 24.0);


	/***Bayesian Decision***/

	Mat XsubMean1 = Mat::zeros(13, 1, CV_32FC1);
	Mat XsubMean2 = Mat::zeros(13, 1, CV_32FC1);
	Mat XsubMean3 = Mat::zeros(13, 1, CV_32FC1);

	double posterior1 = 0.0;
	double posterior2 = 0.0;
	double posterior3 = 0.0;
	double totalerror = 0.0;

	int error1 = 0;
	int error2 = 0;
	int error3 = 0;


	for (int i = 0; i < 29; i++)
	{
		for (int j = 0; j < 13; j++)
		{
			XsubMean1.at<float>(j, 0) = matrix_a1_test.at<float>(j, i) - Mean_a1_train_tp1.at<float>(j, 0);
			XsubMean2.at<float>(j, 0) = matrix_a1_test.at<float>(j, i) - Mean_a2_train_tp1.at<float>(j, 0);
			XsubMean3.at<float>(j, 0) = matrix_a1_test.at<float>(j, i) - Mean_a3_train_tp1.at<float>(j, 0);
			//printf("%f ", XsubMean1.at<float>(j, 0));
		}
		/***posterior1***/
		BayesianClassifer(XsubMean1, Covar_a1_2_train, prior_class1, posterior1, 1, 13, 13, 13);

		/***posterior2***/
		BayesianClassifer(XsubMean2, Covar_a2_2_train, prior_class2, posterior2, 1, 13, 13, 13);

		/***posterior3***/
		BayesianClassifer(XsubMean3, Covar_a3_2_train, prior_class3, posterior3, 1, 13, 13, 13);

		/***find Minimum of Posteriors***/
		MinPosterior(posterior1, posterior2, posterior3, posterior1, error1);

	}
	printf("error1: %d\n", error1);
	if (result_of_classification.is_open())
	{
		result_of_classification << "count of error1: " << error1 << endl;
	}


	for (int i = 0; i < 35; i++)
	{
		for (int j = 0; j < 13; j++)
		{
			XsubMean1.at<float>(j, 0) = matrix_a2_test.at<float>(j, i) - Mean_a1_train_tp1.at<float>(j, 0);
			XsubMean2.at<float>(j, 0) = matrix_a2_test.at<float>(j, i) - Mean_a2_train_tp1.at<float>(j, 0);
			XsubMean3.at<float>(j, 0) = matrix_a2_test.at<float>(j, i) - Mean_a3_train_tp1.at<float>(j, 0);
			//printf("%f ", XsubMean1.at<float>(j, 0));
		}
		/***posterior1***/
		BayesianClassifer(XsubMean1, Covar_a1_2_train, prior_class1, posterior1, 1, 13, 13, 13);

		/***posterior2***/
		BayesianClassifer(XsubMean2, Covar_a2_2_train, prior_class2, posterior2, 1, 13, 13, 13);

		/***posterior3***/
		BayesianClassifer(XsubMean3, Covar_a3_2_train, prior_class3, posterior3, 1, 13, 13, 13);

		/***find Minimum of Posteriors***/
		MinPosterior(posterior1, posterior2, posterior3, posterior2, error2);

	}
	printf("error2: %d\n", error2);
	if (result_of_classification.is_open())
	{
		result_of_classification << "count of error2: " << error2 << endl;
	}

	for (int i = 0; i < 24; i++)
	{
		for (int j = 0; j < 13; j++)
		{
			XsubMean1.at<float>(j, 0) = matrix_a3_test.at<float>(j, i) - Mean_a1_train_tp1.at<float>(j, 0);
			XsubMean2.at<float>(j, 0) = matrix_a3_test.at<float>(j, i) - Mean_a2_train_tp1.at<float>(j, 0);
			XsubMean3.at<float>(j, 0) = matrix_a3_test.at<float>(j, i) - Mean_a3_train_tp1.at<float>(j, 0);
			//printf("%f ", XsubMean1.at<float>(j, 0));
		}
		/***posterior1***/
		BayesianClassifer(XsubMean1, Covar_a1_2_train, prior_class1, posterior1, 1, 13, 13, 13);

		/***posterior2***/
		BayesianClassifer(XsubMean2, Covar_a2_2_train, prior_class2, posterior2, 1, 13, 13, 13);

		/***posterior3***/
		BayesianClassifer(XsubMean3, Covar_a3_2_train, prior_class3, posterior3, 1, 13, 13, 13);
		/***find Minimum of Posteriors***/
		MinPosterior(posterior1, posterior2, posterior3, posterior3, error3);
	}
	printf("error3: %d\n", error3);
	if (result_of_classification.is_open())
	{
		result_of_classification << "count of error3: " << error3 << endl;
	}

	totalerror = (error1 + error2 + error3) / 88.0;
	printf("Total Error Rate: %f\n", totalerror);
	if (result_of_classification.is_open())
	{
		result_of_classification << "Total Error Rate: " << totalerror << endl;
	}
	result_of_classification.close();

	return 0;
}
