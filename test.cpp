#include <cv.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

//using namespace cv::gpu;
//using namespace cv;
using namespace std;

int main(int argc, char** argv){
	int cuda = cv::gpu::getCudaEnabledDeviceCount();
	cout << cuda << endl;
//	cout << "Num processors is " << cv::gpu::DeviceInfo::multiProcessorCount() << endl;
//	cout << cv::gpu::DeviceInfo::isCompatible() << endl;
	cv::gpu::DeviceInfo info;
	cout << info.multiProcessorCount() << endl;
	cv::gpu::setDevice(0);
	cv::gpu::GpuMat aa;
	
	cv::Mat src_host = cv::imread("img.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::gpu::GpuMat dst, src;
	src.upload(src_host);

	cv::gpu::threshold(src,dst, 128.0, 255.0, CV_THRESH_BINARY);

	cv::Mat result_host(dst);
	cv::imshow("Result", result_host);
	cv::waitKey();
}
