#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <sys/time.h>

//using namespace cv::gpu;
//using namespace cv;
using namespace std;

int main(int argc, char** argv){

	timeval tim;
 
 
	int cuda = cv::gpu::getCudaEnabledDeviceCount();
	cout << cuda << endl;
//	cout << "Num processors is " << cv::gpu::DeviceInfo::multiProcessorCount() << endl;
//	cout << cv::gpu::DeviceInfo::isCompatible() << endl;
	cout << "Initializing Cuda?\n";
	cv::gpu::DeviceInfo info;
	cout << info.multiProcessorCount() << endl;
	cv::gpu::setDevice(0);
	cv::gpu::GpuMat aa;
	
	cv::Mat src_host = cv::imread("city_1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::gpu::GpuMat dst, src;
	cout <<"Uploading\n";
	src.upload(src_host);

	cout << "threshold\n";
	cv::gpu::threshold(src,dst, 128.0, 255.0, CV_THRESH_BINARY);

	cv::Mat result_host(dst);
	cout << "Obtained result. Now uploading again.\n";
	cv::gpu::GpuMat d2, s2;
	cout << "Upload #2\n";
             gettimeofday(&tim, NULL);
             double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	s2.upload(src_host);
for (int i = 0; i < 5000; i++){
	cv::gpu::threshold(s2,d2,128,255,CV_THRESH_BINARY);
}
             gettimeofday(&tim, NULL);
             double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
             cout<<t2-t1<<" seconds elapsed\n";
	
	cv::Mat rhost2(d2);
	cout << "Threshold #2 done\n";

	cv::Mat noGPUdst;
             gettimeofday(&tim, NULL);
             double t3=tim.tv_sec+(tim.tv_usec/1000000.0);
for (int i = 0; i < 5000; i++)
	cv::threshold(src_host, noGPUdst, 128,255, CV_THRESH_BINARY);

             gettimeofday(&tim, NULL);
             double t4=tim.tv_sec+(tim.tv_usec/1000000.0);
             cout<<t4-t3<<" seconds elapsed\n";
//	cv::imshow("Result", result_host);
//	cv::waitKey();
}
