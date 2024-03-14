#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;

//��˹ƽ��
Mat gaussianBlur(Mat src, int ksize, double sigma);

//����ָ�
void textureSegmentation(const Mat& inputImage, int blockSize);

//�Ҷȷָ�
void grayScaleSegmentation(Mat& src, Mat& dst, int threshold);

//ͨ����Ե���ػҽ׷ָ���ֵ
int getThresholdValue(const cv::Mat& image);

//����ͼ��ĻҶȹ�������
Mat grayLevelCoOccurrenceMatrix(Mat image, int d, int theta, int levels);

//����Ҷȹ�������ĶԱȶ�
float contrast(Mat coOccurrenceMatrix);

//Stentifordϸ���㷨
Mat stentifordThinning(Mat inputImage);

//��¼ͼ����˶��켣
void show_motion_trajectory(vector<Mat> frames, vector<Point2f> trajectory,int count);

//Ԥ��ͼ����˶��ٶ�
void predictMotionSpeed(vector<Mat>& frames);

double getContourArea(Mat& img, int threshold,int count);