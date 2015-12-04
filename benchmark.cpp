#include <iostream>
#include "opencv2/opencv.hpp"
#include <sys/time.h>
#include <opencv2/nonfree/nonfree.hpp>

//#include "surf.cu"

#define GPU 1
#define CPU !GPU

struct timingStruct{
	double detector_time;
	double extractor_time;
	double matcher_time;
	double findFundMat_time;
	unsigned int frameNum;
};

#if GPU
	#include "opencv2/gpu/gpu.hpp"
	#include "opencv2/nonfree/gpu.hpp"
	//using namespace cv::gpu;
#endif

using namespace cv;
using namespace std;

inline double time(timeval *tim){
	gettimeofday(tim, NULL);
	return (tim->tv_sec + (tim->tv_usec/1000000.0) );
}

int main(int argc, char** argv){
	
	if (argc < 2){
		cout << "Insufficient args; usage ./bench <input file>\n";
		exit(0);
	}
	
	timeval tim;
	int thresh = 7.0;
	Mat frame;
	
#if GPU
	cout << "Initializing Cuda\n";
	cv::gpu::DeviceInfo info;
	cv::gpu::setDevice(0);
	gpu::FAST_GPU gpuFastDetector(thresh);
	gpu::SURF_GPU surf(thresh, 4, 2, true, 0.01f, false);
	
	cv::Mat src_host = cv::imread("city_1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::gpu::GpuMat dst1, src1;
	src1.upload(src_host);
	cv::gpu::threshold(src1,dst1, 128.0, 255.0, CV_THRESH_BINARY);
	src1.release();
	dst1.release();

	vector< vector < timingStruct > > aggregateGPU_times;
	
//TODO: Decide where to put the start timer.
	for (int i = 0; i < 1; i++){
		// Stats info
		double total = 0.0;
		double max = 0.0;
		double detector_total = 0.0;
		double extractor_total = 0.0;
		double min = 100.0;
		bool started = false;
		int numFrames = 0;
				
		vector<timingStruct> GPU_times;
		
		// Open the video and catch if it's not a video.
		VideoCapture capture(argv[1]);
		if (!capture.isOpened() )
			throw "Error when opening video file.\n";
		cout << "Start loop GPU\n";

		vector<float> test;
		int ii = 0;
		gpu::GpuMat descriptors, descriptors_keyframe, GPU_keypoints, GPU_first_keypoints; // Have to be in scope
		for ( ; ; ){
			double start, detector_end, extractor_end, matcherStart, matcherEnd, fundamentalMatEnd;
			ii++;
			capture >> frame;
			if (frame.empty() ){
				cout << " Break\n";
				break;
			}
			start = time(&tim);
			cvtColor(frame, frame, COLOR_BGR2GRAY);
			gpu::GpuMat src;//kps, src, empty;
			src.upload(frame);
			gpuFastDetector(src, gpu::GpuMat(), GPU_keypoints);
			detector_end = time(&tim);
			
			const int nFeatures = GPU_keypoints.cols;
			
			vector<KeyPoint> vecKps;
			gpuFastDetector.downloadKeypoints(GPU_keypoints, vecKps);
			surf.uploadKeypoints(vecKps, GPU_keypoints);
			
			surf(src, gpu::GpuMat(), GPU_keypoints, descriptors, true);
			extractor_end = time(&tim);
			cout << "GPU descriptors size is " << descriptors.cols << " " <<  descriptors.rows << endl;
			
			if (!started){
				started = true;
				GPU_keypoints.copyTo(GPU_first_keypoints);
				descriptors.copyTo(descriptors_keyframe);
			}else{ // BF Matching
				matcherStart = time(&tim);
				vector< vector< DMatch > > doubleMatches;
			//	vector< DMatch > matches;
				vector< DMatch > good_matches;
				gpu::BruteForceMatcher_GPU< L2<float> > matcher;
				
			//	BFMatcher matcher;
				matcher.knnMatch( descriptors, descriptors_keyframe, doubleMatches, 2 );
				
				/*for (int i = 0; i < doubleMatches.size(); i++) {
					cout << doubleMatches[i][0].distance << "/" << doubleMatches[i][1].distance << ", ";
				}
				cout << endl;*/
				
				double ratio = 0.8;
				for (int i = 0; i < doubleMatches.size(); i++) {
					if (doubleMatches[i][0].distance < ratio * 
						doubleMatches[i][1].distance){
					good_matches.push_back(doubleMatches[i][0]);
					}
				}
				
				vector<Point2d> matched_kps_moved, matched_kps_keyframe;
				vector<KeyPoint> keypoints, first_keypoints;
				
				for( int i = 0; i < good_matches.size(); i++ ){
				  matched_kps_moved.push_back( keypoints[ good_matches[i].queryIdx ].pt );  // Left frame
				  matched_kps_keyframe.push_back( first_keypoints[ good_matches[i].trainIdx ].pt );
				}
				matcherEnd = time(&tim);
							
				if (! (matched_kps_moved.size() < 4 || matched_kps_keyframe.size() < 4) ){
					std::vector<uchar> status; 
	
					double fMatP1 = 1.0;
					double fMatP2 = 0.995;
		
				// Use RANSAC and the fundamental matrix to take out points that don't fit geometrically
					findFundamentalMat(matched_kps_moved, matched_kps_keyframe,
						CV_FM_RANSAC, fMatP1, fMatP2, status);
				}
				
				fundamentalMatEnd = time(&tim);
								
			}
			
			double end = time(&tim);	
			double elapsed = end - start;
			
			timingStruct tmp;
			tmp.detector_time = detector_end - start;
			tmp.extractor_time = extractor_end - detector_end;
			tmp.matcher_time = matcherEnd - matcherStart; 
			tmp.findFundMat_time = fundamentalMatEnd - matcherEnd;
			tmp.frameNum = numFrames++;
			GPU_times.push_back(tmp);
			
			cout << "Elapsed = " << elapsed << endl;
			
			src.release(); // Release the memory for the GPU.
			GPU_keypoints.release();
			
		}
		
		aggregateGPU_times.push_back(GPU_times);
	}        
        
#endif

#if(CPU)
	FastFeatureDetector detector(thresh);
	SurfDescriptorExtractor extractor; 
	
	vector< vector < timingStruct > > aggregateCPU_times;

	for (int i = 0; i < 1; i++){
		double total = 0.0;
		double detector_total = 0.0;
		double extractor_total = 0.0;
		double max = 0.0;
		double min = 100.0;
		bool started = false;
		int numFrames = 0;
		VideoCapture capture(argv[1]);
		if (!capture.isOpened() )
			throw "Error when opening video file.\n";
		cout << "Start loop CPU\n";
		Mat descriptors, descriptors_keyframe; // Have to be in scope
		vector<KeyPoint> keypoints, first_keypoints;
		
		vector<timingStruct> CPU_times;
		
		for ( ; ; ){
			capture >> frame;
			if (frame.empty() )
				break;
			double start = time(&tim);
			cvtColor(frame, frame, COLOR_BGR2GRAY);
			detector.detect(frame, keypoints);
			double detector_end = time(&tim);
			extractor.compute(frame, keypoints, descriptors); 
			double extractor_end = time(&tim);
			double matcherStart, matcherEnd, fundamentalMatEnd;
			if (!started){
				started = true;
				first_keypoints = keypoints;
				descriptors.copyTo(descriptors_keyframe);
			}else{ // BF Matching
				matcherStart = time(&tim);
				vector< vector< DMatch > > doubleMatches;
				vector< DMatch > matches;
				vector< DMatch > good_matches;
				BFMatcher matcher;
				matcher.knnMatch( descriptors, descriptors_keyframe, doubleMatches, 2 );
				
				
			/*	for (int i = 0; i < doubleMatches.size(); i++) {
					cout << doubleMatches[i][0].distance << "/" <<  doubleMatches[i][1].distance << ", ";
				}
				cout << endl;*/
				
				double ratio = 0.8;
				for (int i = 0; i < doubleMatches.size(); i++) {
					if (doubleMatches[i][0].distance < ratio * 
						doubleMatches[i][1].distance){
					good_matches.push_back(doubleMatches[i][0]);
					}
				}
			
				vector<Point2d> matched_kps_moved, matched_kps_keyframe;
				for( int i = 0; i < good_matches.size(); i++ ){
				  matched_kps_moved.push_back( keypoints[ good_matches[i].queryIdx ].pt );  // Left frame
				  matched_kps_keyframe.push_back( first_keypoints[ good_matches[i].trainIdx ].pt );
				}
				
				matcherEnd = time(&tim);
				
				if (! (matched_kps_moved.size() < 4 || matched_kps_keyframe.size() < 4) ){
					std::vector<uchar> status; 
	
					double fMatP1 = 1.0;
					double fMatP2 = 0.995;
		
				// Use RANSAC and the fundamental matrix to take out points that don't fit geometrically
					findFundamentalMat(matched_kps_moved, matched_kps_keyframe,
						CV_FM_RANSAC, fMatP1, fMatP2, status);
				}
				fundamentalMatEnd = time(&tim);
			}
			double end = time(&tim);	
			double elapsed = end - start;
			
			timingStruct tmp;
			tmp.detector_time = detector_end - start;
			tmp.extractor_time = extractor_end - detector_end;
			tmp.matcher_time = matcherEnd - matcherStart; 
			tmp.findFundMat_time = fundamentalMatEnd - matcherEnd;
			tmp.frameNum = numFrames++;
			CPU_times.push_back(tmp);
			
			cout << "Elapsed = " << elapsed << endl;
			/*if (elapsed > max){
				max = elapsed;
			}
			if (elapsed < min){
				min = elapsed;
			}
			total += elapsed;
			detector_total += detector_time;
			extractor_total += extractor_time;
			numFrames++;*/
		}
		aggregateCPU_times.push_back(CPU_times);
		/*cout << "Total: " << total << " Max: " << max << " Min: " << min << " Average: " 
			<< total / numFrames << " Detector_only avg: " << detector_total / numFrames
			<< " Extractor_only avg: " << extractor_total / numFrames<< endl;*/
	}
#endif

	return 0;
}

