#include"hog.h"

cv::Mat Flattening(cv::Mat src) {
	cv::Mat flatsrc(src.rows, src.cols, CV_8U);

	int max = 0, min = 255;
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			if (max < src.at<uchar>(y, x)) max = src.at<uchar>(y, x);
			if (min > src.at<uchar>(y, x)) min = src.at<uchar>(y, x);
		}
	}

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			flatsrc.at<uchar>(y, x) = src.at<uchar>(y, x) * (max - min) / 255;
		}
	}

	return flatsrc;
}

void InitVector(std::vector<std::vector<std::vector<double>>> &vector,
	int cols,
	int rows,
	const int GRADIENT_SIZE) {
	vector.resize(rows);
	for (int j = 0; j < vector.size(); j++) {
		vector[j].resize(cols);
	}
	for (int j = 0; j < vector.size(); j++) {
		for (int i = 0; i < vector[j].size(); i++) {
			vector[j][i].resize(GRADIENT_SIZE);
		}
	}
}

void CalcHistogram(cv::Mat src,
	const std::vector<int> CELL_SIZE,
	const int GRADIENT_SIZE,
	std::vector<std::vector<std::vector<double>>> &histogram) {
	for (int y = 0; y < src.rows; y += CELL_SIZE[1]) {
		for (int x = 0; x < src.cols; x += CELL_SIZE[0]) {
			for (int j = 0; j < CELL_SIZE[1]; j++) {
				for (int i = 0; i < CELL_SIZE[0]; i++) {
					if (y + j - 1 >= 0 && y + j + 1 < src.rows && x + i - 1 >= 0 && x + i + 1 < src.cols) {
						double dx = src.at<uchar>(y + j, x + i + 1) - src.at<uchar>(y + j, x + i - 1);
						double dy = src.at<uchar>(y + j + 1, x + i) - src.at<uchar>(y + j - 1, x + i);

						double theta = std::atan2(dy, dx);
						if (theta < 0.0) theta += PI;
						int orientation = int(theta / PI * double(GRADIENT_SIZE));
						if (orientation == 9) orientation = 0;
						double magnitude = sqrt(dx * dx + dy * dy);

						histogram[y / CELL_SIZE[1]][x / CELL_SIZE[0]][orientation] += magnitude;
					}
				}
			}
		}
	}
}

void CalcHOG(cv::Mat src,
	const std::vector<int> CELL_SIZE,
	const std::vector<int> BLOCK_SIZE,
	const int GRADIENT_SIZE,
	std::vector<std::vector<std::vector<double>>> &hog_vector
	//,cv::Mat &dst
	) 
{
	int cell_cols = src.cols / CELL_SIZE[0] + 1;
	int cell_rows = src.rows / CELL_SIZE[1] + 1;
	int block_cols = cell_cols / BLOCK_SIZE[0] + 1;
	int block_rows = cell_rows / BLOCK_SIZE[1] + 1;

	std::vector<std::vector<std::vector<double>>> histogram;
	std::vector<std::vector<std::vector<double>>> sum;
	InitVector(histogram, cell_cols, cell_rows, GRADIENT_SIZE);
	InitVector(sum, block_cols, block_rows, GRADIENT_SIZE);


	CalcHistogram(src, CELL_SIZE, GRADIENT_SIZE, histogram);

	for (int y = 0; y < cell_rows; y += BLOCK_SIZE[1]) {
		for (int x = 0; x < cell_cols; x += BLOCK_SIZE[0]) {
			for (int j = 0; j < BLOCK_SIZE[1]; j++) {
				for (int i = 0; i < BLOCK_SIZE[0]; i++) {
					for (int bin = 0; bin < GRADIENT_SIZE; bin++) {
						if (y + j < cell_rows && x + i < cell_cols)
							sum[y / BLOCK_SIZE[1]][x / BLOCK_SIZE[0]][bin] += pow(histogram[bin][y + j][x + i], 2);
					}
				}
			}

			for (int j = 0; j < BLOCK_SIZE[1]; j++) {
				for (int i = 0; i < BLOCK_SIZE[0]; i++) {
					for (int bin = 0; bin < GRADIENT_SIZE; bin++) {
						if (y + j < cell_rows && x + i < cell_cols)
							hog_vector[y + j][x + i][bin] = histogram[y + j][x + i][bin] / sqrt(sum[y / BLOCK_SIZE[1]][x / BLOCK_SIZE[0]][bin] + epsilon);
					}
				}
			}

			/*for (int j = 0; j < BLOCK_SIZE[1]; j++) {
				for (int i = 0; i < BLOCK_SIZE[0]; i++) {
					cv::Point center = cv::Point((x + i) * CELL_SIZE[0], (y + j) * CELL_SIZE[1]) + cv::Point(CELL_SIZE[0] / 2, CELL_SIZE[1] / 2);
					for (int bin = 0; bin < GRADIENT_SIZE; bin++) {
						if (y + j < cell_rows && x + i < cell_cols) {
							if (hog_vector[y + j][x + i][bin] > hog_th) {
								double rad = (double(bin * 180 / GRADIENT_SIZE) + 90.0) * PI / 180.0;
								cv::Point rd(double(CELL_SIZE[0]) * 0.5 * cos(rad), double(CELL_SIZE[1]) * 0.5 * sin(rad));
								cv::Point rp = center - rd;
								cv::Point lp = center + rd;
								cv::line(dst, rp, lp, cv::Scalar(255 * hog_vector[y + j][x + i][bin]));
							}
						}
					}
				}
			}*/
		}
	}
}