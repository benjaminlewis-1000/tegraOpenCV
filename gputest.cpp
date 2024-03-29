#include <iostream>
#include "opencv2/opencv.hpp"
#include <sys/time.h>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/nonfree/gpu.hpp"
#include "HomogCUDA/CUDA_RANSAC_Homography.h"

using namespace cv;
using namespace std;

inline double time(timeval *tim){
	gettimeofday(tim, NULL);
	return (tim->tv_sec + (tim->tv_usec/1000000.0) );
}

//TODO: Figure out the order that they should be put into the matcher.

int main(int argc, char** argv){

	int thresh = 25.0;
	timeval tim;
	
	cout << "Initializing Cuda\n";
	cv::gpu::DeviceInfo info;
	cv::gpu::setDevice(0);
	gpu::FAST_GPU gpuFastDetector(thresh);
	gpu::SURF_GPU surf(thresh, 4, 2, true, 0.01f, false);
		
	cv::Mat img1 = cv::imread("img2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat img2 = cv::imread("img.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	int numFrames = 0;

	gpu::GpuMat src1, src2;//kps, src, empty;
	// Upload the two images to GPU memory.
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
	
	// Calculate the SURF features and put them in desc1/desc2.
	surf(src1, gpu::GpuMat(), kps1, desc1, true);
	surf(src2, gpu::GpuMat(), kps2, desc2, true);
	
	// Show how many descriptors were found in image 1 and 2 respectively.
	cout << "GPU descriptors size is ~" << desc1.rows << " and " <<  desc2.rows << endl;
	
	vector< vector< DMatch > > doubleMatches;
	vector< DMatch > good_matches;
	gpu::BruteForceMatcher_GPU< L2<float> > matcher;
	
//	BFMatcher matcher;
	matcher.knnMatch( desc1, desc2, doubleMatches, 2 );
	
/*	cout << "Matches found: " << doubleMatches.size() << endl;
	for (int i = 0; i < doubleMatches.size(); i++) {
		if (doubleMatches[i][0].distance != 0)
			cout << doubleMatches[i][0].distance << "/" << doubleMatches[i][1].distance << ", ";
	}
	cout << endl;
*/

	// Compare the matches. If the first two matches are two close, they aren't good matches. 	
	double ratio = 0.8;
	for (int i = 0; i < doubleMatches.size(); i++) {
		if (doubleMatches[i][0].distance < ratio * 
			doubleMatches[i][1].distance){
		good_matches.push_back(doubleMatches[i][0]);
		}
	}
	
	cout << "There are " << good_matches.size() << " good matches." << endl;
	
	vector<Point2d> matched_kps_moved, matched_kps_keyframe;
	
	cout << "Pushing back..." << endl;
	cout << "Keypoint sizes: " << vecKps1.size() << ", " << vecKps2.size() << endl;
	
	// Find the location of the points in the original vectors, as indexed by good_matches
	// train and query indices. Push them back onto two vectors which can be used in 
	// findFundamentalMat.
	if (good_matches.size() > 0){
		for( int i = 0; i < good_matches.size(); i++ ){
		//cout << good_matches[i].queryIdx << ", " << good_matches[i].trainIdx << endl;
		  matched_kps_moved.push_back( vecKps1[ good_matches[i].queryIdx ].pt );  // Left frame
		  matched_kps_keyframe.push_back( vecKps2[ good_matches[i].trainIdx ].pt );
		}
	}
	cout << matched_kps_moved.size() << ", " << matched_kps_keyframe.size() << endl;
	
	double fmStart = time(&tim);
	if (! (matched_kps_moved.size() < 4 || matched_kps_keyframe.size() < 4) ){
		std::vector<uchar> status; 

		double fMatP1 = 1.0;
		double fMatP2 = 0.995;

	// Use RANSAC and the fundamental matrix to take out points that don't fit geometrically
		findFundamentalMat(matched_kps_moved, matched_kps_keyframe,
			CV_FM_RANSAC, fMatP1, fMatP2, status);
	}
	double fmEnd = time(&tim);
	cout << "Fund. mat time is " << fmEnd - fmStart << endl;

	
	return 0;
}

