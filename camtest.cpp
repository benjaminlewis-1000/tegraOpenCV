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
	if (argc < 1){
		cout << "Insufficient args; usage ./cam <input image>" << endl;
		exit(0);
	}

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
	
	int thresh = 5;
	
	FastFeatureDetector detector(thresh);
	gpu::FAST_GPU d2(thresh);
	
	for (;;){
		Mat img;
		cap >> img;
	cout << img.channels() << " channels and " << src.channels() << " channels. "<< endl;
		if (img.rows > 0){
			gpu::GpuMat frame, dst;
			cvtColor(img, img, COLOR_BGR2GRAY);
             double t1=time(&tim);
			frame.upload(img);
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
			
			double fastStart = time(&tim);
			vector<KeyPoint> keypoints;
			detector.detect(img, keypoints);
			double fastEnd = time(&tim);
			cout << "Fast time is " << fastEnd - fastStart << endl;
				gpu::GpuMat in, kps;
				in.upload(img);
			fastStart = time(&tim);
				d2(in, kps, kps);
			fastEnd = time(&tim);
			cout << "Fast time is " << fastEnd - fastStart << endl << endl;

		}
	} 

}
