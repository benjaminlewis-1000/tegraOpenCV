/*
Copyright 2011 Nghia Ho. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY NGHIA HO ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL BY NGHIA HO OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Nghia Ho.
*/

#include <sys/time.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>

#include "CUDA_RANSAC_Homography.h"

using namespace std;
using namespace cv;

// Calc the theoretical number of iterations using some conservative parameters
const double CONFIDENCE = 0.99;
const double INLIER_RATIO = 0.18; // Assuming lots of noise in the data!
const double INLIER_THRESHOLD = 3.0; // pixel distance

double TimeDiff(timeval t1, timeval t2)
{
    double t;
    t = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    t += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

    return t;
}

int main(int argc, char **argv)
{
    if(argc != 4) {
        printf("Usage: CUDA_RANSAC_Homography [img.jpg] [target.jpg] [results.png]\n");
        return 0;
    }

    timeval start_time, t1, t2;
    Mat img1, img2;
    vector<KeyPoint> kp1, kp2;
    Mat grey1, grey2;
    int best_inliers;
    float best_H[9];
    vector <char> inlier_mask;
    vector <Point2Df> src, dst;
    vector <float> match_score;
    int K;
    int opencv_inliers;

    assert(cv::gpu::getCudaEnabledDeviceCount());

    gettimeofday(&start_time, NULL);

    printf("--------------------------------\n");

    // Load images
    {
        gettimeofday(&t1, NULL);

        img1 = imread(argv[1]);
        img2 = imread(argv[2]);

        assert(img1.data);
        assert(img2.data);

        gettimeofday(&t2, NULL);
    }
    printf("Load images: %g ms\n", TimeDiff(t1,t2));

    // Convert to greyscale
    {
        gettimeofday(&t1, NULL);

        cv::cvtColor(img1, grey1, CV_BGR2GRAY);
        cv::cvtColor(img2, grey2, CV_BGR2GRAY);

        gettimeofday(&t2, NULL);
    }
    printf("Convert to greyscale: %g ms\n", TimeDiff(t1,t2));

    // SURF
    {
        gettimeofday(&t1, NULL);


        cv::gpu::SURF_GPU surf(100, 4, 2, true, 0.01f, false);
        cv::gpu::GpuMat gpu_grey1;
        gpu_grey1.upload(grey1);
        cv::gpu::GpuMat gpu_grey2;
        gpu_grey2.upload(grey2);
        cv::gpu::GpuMat gpu_kp1, gpu_kp2;
        cv::gpu::GpuMat gpu_desc1, gpu_desc2;
        cv::gpu::GpuMat gpu_ret_idx, gpu_ret_dist, gpu_all_dist;
        cv::Mat ret_idx, ret_dist;

        printf("Size of grey1 is %d\n", gpu_grey1.cols);

        printf("Surf starting...\n");
        surf(gpu_grey2, cv::gpu::GpuMat(), gpu_kp2, gpu_desc2);
        printf("Surf 2\n");
        surf(gpu_grey1, cv::gpu::GpuMat(), gpu_kp1, gpu_desc1);

        surf.downloadKeypoints(gpu_kp1, kp1);
        surf.downloadKeypoints(gpu_kp2, kp2);

        printf("Surf done\n");

        cv::gpu::BruteForceMatcher_GPU < cv::L2<float> > gpu_matcher;

        gpu_matcher.knnMatchSingle(gpu_desc1, gpu_desc2, gpu_ret_idx, gpu_ret_dist, gpu_all_dist, 2);

        gpu_ret_idx.download(ret_idx);
        gpu_ret_dist.download(ret_dist);

        float ratio = 0.7f;
        float min_val = FLT_MAX;
        float max_val = 0.0f;

        for(int i=0; i < ret_idx.rows; i++) {
            if(ret_dist.at<float>(i,0) < ret_dist.at<float>(i,1)*ratio) {
                int idx = ret_idx.at<int>(i,0);

                Point2Df a, b;

                a.x = kp1[i].pt.x;
                a.y = kp1[i].pt.y;

                b.x = kp2[idx].pt.x;
                b.y = kp2[idx].pt.y;

                src.push_back(a);
                dst.push_back(b);
                match_score.push_back(ret_dist.at<float>(i,0));

                if(ret_dist.at<float>(i,0) < min_val) {
                    min_val = ret_dist.at<float>(i,0);
                }

                if(ret_dist.at<float>(i,0) > max_val) {
                    max_val = ret_dist.at<float>(i,0);
                }
            }
        }

        // Flip score
        for(unsigned int i=0; i < match_score.size(); i++) {
            match_score[i] = max_val - match_score[i] + min_val;
        }

        gettimeofday(&t2, NULL);
    }
    printf("GPU SURF: %g ms\n", TimeDiff(t1,t2));

    // OpenCV homography
   /* {
        gettimeofday(&t1, NULL);

        Mat src2(src.size(), 2, CV_32F);
        Mat dst2(dst.size(), 2, CV_32F);

        for(unsigned int i=0; i < src.size(); i++) {
            src2.at<float>(i,0) = src[i].x;
            src2.at<float>(i,1) = src[i].y;
        }

        for(unsigned int i=0; i < dst.size(); i++) {
            dst2.at<float>(i,0) = dst[i].x;
            dst2.at<float>(i,1) = dst[i].y;
        }

        vector<uchar> status;
        findHomography(src2, dst2,status, CV_RANSAC, INLIER_THRESHOLD);
        opencv_inliers = accumulate(status.begin(), status.end(), 0);

        gettimeofday(&t2, NULL);
    }
    printf("RANSAC Homography (OpenCV): %g ms\n", TimeDiff(t1,t2));*/

    // Homography
    {
        gettimeofday(&t1, NULL);

        K = (int)(log(1.0 - CONFIDENCE) / log(1.0 - pow(INLIER_RATIO, 4.0)));

        CUDA_RANSAC_Homography(src, dst, match_score, INLIER_THRESHOLD, K, &best_inliers, best_H, &inlier_mask);

        gettimeofday(&t2, NULL);
    }
    printf("RANSAC Homography (GPU): %g ms\n", TimeDiff(t1,t2));

    // Refine homography
    {
        gettimeofday(&t1, NULL);

        Mat src2(best_inliers, 2, CV_32F);
        Mat dst2(best_inliers, 2, CV_32F);

        int k = 0;
        for(unsigned int i=0; i < src.size(); i++) {
            if(inlier_mask[i] == 0) {
                continue;
            }

            src2.at<float>(k,0) = src[i].x;
            src2.at<float>(k,1) = src[i].y;

            dst2.at<float>(k,0) = dst[i].x;
            dst2.at<float>(k,1) = dst[i].y;

            k++;
        }

        vector<uchar> status;
        Mat refined_H = findHomography(src2, dst2, status, 0 /* Least square */);

         k =0;
        for(int y=0; y < 3; y++) {
            for(int x=0; x < 3; x++) {
                best_H[k] = refined_H.at<double>(y,x);
                k++;
            }
        }

        best_inliers = 0;
        for(int i=0; i < src.size(); i++) {
            float x = best_H[0]*src[i].x + best_H[1]*src[i].y + best_H[2];
            float y = best_H[3]*src[i].x + best_H[4]*src[i].y + best_H[5];
            float z = best_H[6]*src[i].x + best_H[7]*src[i].y + best_H[8];

            x /= z;
            y /= z;

            float dist_sq = (dst[i].x - x)*(dst[i].x- x) + (dst[i].y - y)*(dst[i].y - y);

            if(dist_sq < INLIER_THRESHOLD) {
               best_inliers++;
            }
        }

        gettimeofday(&t2, NULL);
    }
    printf("Refine homography: %g ms\n", TimeDiff(t1,t2));

    printf("--------------------------------\n");
    printf("Total time: %g ms\n", TimeDiff(start_time,t2));

    printf("\n");
    printf("Features extracted: %d %d\n", kp1.size(), kp2.size());
    printf("OpenCV Inliers: %d\n", opencv_inliers);
    printf("GPU Inliers: %d\n", best_inliers);
    printf("GPU RANSAC iterations: %d\n", K);
    printf("\n");
    printf("RANSAC parameters:\n");
    printf("Confidence: %g\n", CONFIDENCE);
    printf("Inliers ratio: %g\n", INLIER_RATIO);
#ifdef NORMALISE_INPUT_POINTS
    printf("Data is normalised: yes\n");
#else
    printf("Data is normalised: no\n");
#endif

#ifdef BIAS_RANDOM_SELECTION
    printf("Bias random selection: yes\n");
#else
    printf("Bias random selection: no\n");
#endif

    printf("\n");
    printf("Homography matrix\n");

    for(int i=0; i < 9; i++) {
        printf("%g ", best_H[i]/best_H[8]);

        if((i+1) % 3 == 0 && i > 0) {
            printf("\n");
        }
    }

    // Display results
    {
        int h = grey1.rows + grey2.rows;
        int w = max(grey1.cols, grey2.cols);

        Mat result(h, w, CV_8UC3);

        for(int y=0; y < grey1.rows; y++) {
            for(int x=0; x < grey1.cols; x++) {
                result.at<Vec3b>(y,x)[0] = grey1.at<uchar>(y,x);
                result.at<Vec3b>(y,x)[1] = grey1.at<uchar>(y,x);
                result.at<Vec3b>(y,x)[2] = grey1.at<uchar>(y,x);
            }
        }

        for(int y=0; y < grey2.rows; y++) {
            for(int x=0; x < grey2.cols; x++) {
                result.at<Vec3b>(y+grey1.rows,x)[0] = grey2.at<uchar>(y,x);
                result.at<Vec3b>(y+grey1.rows,x)[1] = grey2.at<uchar>(y,x);
                result.at<Vec3b>(y+grey1.rows,x)[2] = grey2.at<uchar>(y,x);        }
        }

        for(unsigned int i=0; i < inlier_mask.size(); i++) {
            if(inlier_mask[i]) {
                line(result, Point(src[i].x, src[i].y), Point(dst[i].x, grey1.rows + dst[i].y), CV_RGB(255,0,0));
            }
        }

        imwrite(argv[3], result);
    }

    return 0;
}
