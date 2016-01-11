#ifndef HOG_h
#define HOG_h

// Header
#include <opencv2/opencv.hpp>
#include <ctype.h>
#include <iostream>
#include <vector>
#include <cmath>

// Global Variables
const double PI = 3.141592;
const std::vector<int> CELL_SIZE = { 8, 8 };
const std::vector<int> BLOCK_SIZE = { 2, 2 };
const int GRADIENT_SIZE = 9; // 0 - 160
const double epsilon = 1.0;
const double hog_th = 0.8;

cv::Mat Flattening(cv::Mat src);

void InitVector(std::vector<std::vector<std::vector<double>>> &vector,
	int cols,
	int rows,
	const int GRADIENT_SIZE);

void CalcHistogram(cv::Mat src,
	const std::vector<int> CELL_SIZE,
	const int GRADIENT_SIZE,
	std::vector<std::vector<std::vector<double>>> &histogram);

void CalcHOG(cv::Mat src,
	const std::vector<int> CELL_SIZE,
	const std::vector<int> BLOCK_SIZE,
	const int GRADIENT_SIZE,
	std::vector<std::vector<std::vector<double>>> &hog_vector
	//,cv::Mat &dst
	);
#endif