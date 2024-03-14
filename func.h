#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;

//高斯平滑
Mat gaussianBlur(Mat src, int ksize, double sigma);

//纹理分割
void textureSegmentation(const Mat& inputImage, int blockSize);

//灰度分割
void grayScaleSegmentation(Mat& src, Mat& dst, int threshold);

//通过边缘像素灰阶分割阈值
int getThresholdValue(const cv::Mat& image);

//计算图像的灰度共生矩阵
Mat grayLevelCoOccurrenceMatrix(Mat image, int d, int theta, int levels);

//计算灰度共生矩阵的对比度
float contrast(Mat coOccurrenceMatrix);

//Stentiford细化算法
Mat stentifordThinning(Mat inputImage);

//记录图像的运动轨迹
void show_motion_trajectory(vector<Mat> frames, vector<Point2f> trajectory,int count);

//预测图像的运动速度
void predictMotionSpeed(vector<Mat>& frames);

double getContourArea(Mat& img, int threshold,int count);