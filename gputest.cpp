#include <iostream>
#include "opencv2/opencv.hpp"
#include <sys/time.h>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/nonfree/gpu.hpp"

using namespace cv;
using namespace std;

inline double time(timeval *tim){
	gettimeofday(tim, NULL);
	return (tim->tv_sec + (tim->tv_usec/1000000.0) );
}

int main(int argc, char** argv){

	int thresh = 100.0;
	timeval tim;
	
	cout << "Initializing Cuda\n";
	cv::gpu::DeviceInfo info;
	cv::gpu::setDevice(0);
	gpu::FAST_GPU gpuFastDetector(thresh);
	gpu::SURF_GPU surf(thresh, 4, 2, true, 0.01f, false);
		
	cv::Mat img1 = cv::imread("img.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat img2 = cv::imread("img2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	int numFrames = 0;

	gpu::GpuMat src1, src2;//kps, src, empty;
	src1.upload(img1);
	src2.upload(img2);
	
	// Calculate FAST features
	gpu::GpuMat kps1, kps2; // Have to be in scope
	gpuFastDetector(src1, gpu::GpuMat(), kps1);
	gpuFastDetector(src2, gpu::GpuMat(), kps2);
		
	// Calculate SURF feature descriptors
	// Have to download the FAST keypoints and reupload them as
	// SURF keypoints in order to get something meaningful. 
	// The overhead is very small for this though.
	gpu::GpuMat desc1, desc2;
	vector<KeyPoint> vecKps1, vecKps2;
	
	gpuFastDetector.downloadKeypoints(kps1, vecKps1);
	gpuFastDetector.downloadKeypoints(kps2, vecKps2);
	surf.uploadKeypoints(vecKps1, kps1);
	surf.uploadKeypoints(vecKps2, kps2);
	
	surf(src1, gpu::GpuMat(), kps1, desc1, true);
	surf(src2, gpu::GpuMat(), kps2, desc2, true);
	vector<float> vecDesc1, vecDesc2;
	surf.downloadDescriptors(desc1, vecDesc1);
	surf.downloadDescriptors(desc2, vecDesc2);
	
	cout << "GPU descriptors size is ~" << desc1.rows << " and " <<  desc2.rows << endl;
	
	/*cout << "Keypoints: " << endl;
	for (int i = 0 ; i < kps.size(); i++){
		cout << kps[i].pt <<  " || " ;
	}
	cout << endl;*/
	
/*	cout << "Descriptors: " << endl;
	for (int i = 0; i < desc.size(); i++){
		cout << desc[i] << " | " ;
	}
	cout << endl;*/
	
	vector< vector< DMatch > > doubleMatches;
	vector< DMatch > matches;
	vector< DMatch > good_matches;
	gpu::BruteForceMatcher_GPU< L2<float> > matcher;
	
//	BFMatcher matcher;
	gpu::GpuMat trainIdx, distance, allDist;
	matcher.knnMatch( desc1, desc2, doubleMatches, 2 );
	
	cout << "Matches found: " << doubleMatches.size() << endl;
	
	for (int i = 0; i < doubleMatches.size(); i++) {
		if (doubleMatches[i][0].distance != 0)
			cout << doubleMatches[i][0].distance << "/" << doubleMatches[i][1].distance << ", ";
	}
	cout << endl;
	
	return 0;
}

