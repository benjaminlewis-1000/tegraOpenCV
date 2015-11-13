#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <sys/time.h>

//using namespace cv::gpu;
//using namespace cv;
using namespace std;
using namespace cv;

inline double time(timeval *tim){
	gettimeofday(tim, NULL);
	return (tim->tv_sec + (tim->tv_usec/1000000.0) );
}

int main(int argc, char** argv){

	timeval tim;
	cv::Mat src = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	/*imshow("Result", src);
	cv::waitKey();*/

	VideoCapture cap(0);
	
	int i = 0;
	double gpu_time = 0.0;
	double max_gpu = 0.0;
	double min_cpu = 1000.0;
	double cpu_time = 0.0;
	
	for (;;){
		Mat img;
		cap >> img;
		if (img.rows > 0){
			gpu::GpuMat frame, dst;
             double t1=time(&tim);
			frame.upload(src);
             double t11=time(&tim);
			gpu::threshold(frame,dst, 128.0, 255.0, CV_THRESH_BINARY);
             double t21=time(&tim);
			cv::Mat result_host(dst);
             double t2=time(&tim);
		cout << "Upload time: " << t11 - t1 << endl;
		//cout << "Gpu time: " << t21 - t11 << endl;
		cout << "Return time: " << t2 - t21 << endl;

	gpu_time += (t21 - t11);

	cv::Mat noGPUdst;
             gettimeofday(&tim, NULL);
             double t3=tim.tv_sec+(tim.tv_usec/1000000.0);
	cv::threshold(src, noGPUdst, 128,255, CV_THRESH_BINARY);

             gettimeofday(&tim, NULL);
             double t4=tim.tv_sec+(tim.tv_usec/1000000.0);
        //     cout<<t4-t3<<" seconds elapsed\n";

	cpu_time += (t4 - t3);
	if (t21 - t11 > max_gpu){
		max_gpu = t21 - t11;
	}
	if (t4 - t3 < min_cpu){
		min_cpu = t4 - t3;
	}
	i++;

	cout << "Average cpu time: " << cpu_time / i << endl
		<< "Min cpu time: " << min_cpu << endl
		<< "Average gpu time: " << gpu_time / i  << endl
		<< "Max gpu time: " << max_gpu << endl;
			//cv::imshow("Result", result_host);
			cv::waitKey(1);

		}
	} 
	/*
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
for (int i = 0; i < 1000; i++){
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
for (int i = 0; i < 1000; i++)
	cv::threshold(src_host, noGPUdst, 128,255, CV_THRESH_BINARY);

             gettimeofday(&tim, NULL);
             double t4=tim.tv_sec+(tim.tv_usec/1000000.0);
             cout<<t4-t3<<" seconds elapsed\n";
//	cv::imshow("Result", result_host);
//	cv::waitKey();*/
}
