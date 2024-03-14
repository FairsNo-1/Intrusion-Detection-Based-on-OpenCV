#include <stdio.h>
#include <math.h>
#include "func.h"
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

//高斯平滑（`src`为输入图像，`ksize`为卷积核大小，`sigma`为高斯核标准差。函数返回经过高斯平滑后的图像。）
Mat gaussianBlur(Mat src, int ksize, double sigma)
{
    Mat dst;
    GaussianBlur(src, dst, Size(ksize, ksize), sigma, sigma);
    return dst;
}

//纹理分割
void textureSegmentation(const Mat& inputImage, int blockSize)
{
    Mat grayImage;
    cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

    int rows = inputImage.rows / blockSize;
    int cols = inputImage.cols / blockSize;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Rect roi(j * blockSize, i * blockSize, blockSize, blockSize);
            Mat block = grayImage(roi);
            imwrite(format("block_%d_%d.jpg", i, j), block);
        }
    }
}

//灰度分割
void grayScaleSegmentation(Mat& src, Mat& dst, int threshold) {
    dst = src.clone();
    int rows = dst.rows;
    int cols = dst.cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int gray = dst.at<uchar>(i, j);
            if (gray < threshold) {
                dst.at<uchar>(i, j) = 0;
            }
            else {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
}


//使用边缘像素计算灰阶分割阈值
int getThresholdValue(const cv::Mat& image) {
    // 计算图像边缘像素的平均灰度值
    int sum = 0;
    int count = 0;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (i == 0 || i == image.rows - 1 || j == 0 || j == image.cols - 1) {
                sum += image.at<uchar>(i, j);
                count++;
            }
        }
    }
    int avg = sum / count;

    // 计算阈值
    int threshold = 0;
    int count1 = 0;
    int count2 = 0;
    int sum1 = 0;
    int sum2 = 0;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image.at<uchar>(i, j) < avg) {
                count1++;
                sum1 += image.at<uchar>(i, j);
            }
            else {
                count2++;
                sum2 += image.at<uchar>(i, j);
            }
        }
    }
    int avg1 = sum1 / count1;
    int avg2 = sum2 / count2;
    threshold = (avg1 + avg2) / 2;

    return threshold;
}

//计算图像的灰度共生矩阵
Mat grayLevelCoOccurrenceMatrix(Mat image, int d, int theta, int levels) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Mat coOccurrenceMatrix = Mat::zeros(levels, levels, CV_32F);

    int dx = d * cos(theta * CV_PI / 180);
    int dy = d * sin(theta * CV_PI / 180);

    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            int x = j + dx;
            int y = i + dy;
            if (x >= 0 && x < gray.cols && y >= 0 && y < gray.rows) {
                int intensity1 = gray.at<uchar>(i, j);
                int intensity2 = gray.at<uchar>(y, x);
                coOccurrenceMatrix.at<float>(intensity1, intensity2)++;
            }
        }
    }

    coOccurrenceMatrix /= sum(coOccurrenceMatrix)[0];
    return coOccurrenceMatrix;
}

//计算灰度共生矩阵的对比度
float contrast(Mat coOccurrenceMatrix) {
    float contrast = 0;
    for (int i = 0; i < coOccurrenceMatrix.rows; i++) {
        for (int j = 0; j < coOccurrenceMatrix.cols; j++) {
            contrast += pow(i - j, 2) * coOccurrenceMatrix.at<float>(i, j);
        }
    }
    return contrast;
}

//Stentiford细化算法
Mat stentifordThinning(Mat inputImage) {
    Mat outputImage = inputImage.clone();
    int rows = outputImage.rows;
    int cols = outputImage.cols;

    // 定义结构元素
    Mat structElement = getStructuringElement(MORPH_CROSS, Size(3, 3));

    // 定义标志位
    bool hasChanged = true;
    while (hasChanged) {
        hasChanged = false;

        // 第一次迭代
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                if (outputImage.at<uchar>(i, j) == 255) {
                    int a = 0;
                    for (int k = -1; k <= 1; k++) {
                        for (int l = -1; l <= 1; l++) {
                            if (outputImage.at<uchar>(i + k, j + l) == 255) {
                                a++;
                            }
                        }
                    }
                    if (a >= 2 && a <= 6) {
                        int b = 0;
                        if (outputImage.at<uchar>(i - 1, j) == 0 && outputImage.at<uchar>(i - 1, j + 1) == 255) b++;
                        if (outputImage.at<uchar>(i - 1, j + 1) == 0 && outputImage.at<uchar>(i, j + 1) == 255) b++;
                        if (outputImage.at<uchar>(i, j + 1) == 0 && outputImage.at<uchar>(i + 1, j + 1) == 255) b++;
                        if (outputImage.at<uchar>(i + 1, j + 1) == 0 && outputImage.at<uchar>(i + 1, j) == 255) b++;
                        if (outputImage.at<uchar>(i + 1, j) == 0 && outputImage.at<uchar>(i + 1, j - 1) == 255) b++;
                        if (outputImage.at<uchar>(i + 1, j - 1) == 0 && outputImage.at<uchar>(i, j - 1) == 255) b++;
                        if (outputImage.at<uchar>(i, j - 1) == 0 && outputImage.at<uchar>(i - 1, j - 1) == 255) b++;
                        if (outputImage.at<uchar>(i - 1, j - 1) == 0 && outputImage.at<uchar>(i - 1, j) == 255) b++;
                        if (b == 1) {
                            outputImage.at<uchar>(i, j) = 0;
                            hasChanged = true;
                        }
                    }
                }
            }
        }

        // 第二次迭代
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                if (outputImage.at<uchar>(i, j) == 255) {
                    int a = 0;
                    for (int k = -1; k <= 1; k++) {
                        for (int l = -1; l <= 1; l++) {
                            if (outputImage.at<uchar>(i + k, j + l) == 255) {
                                a++;
                            }
                        }
                    }
                    if (a >= 2 && a <= 6) {
                        int b = 0;
                        if (outputImage.at<uchar>(i - 1, j) == 0 && outputImage.at<uchar>(i - 1, j + 1) == 255) b++;
                        if (outputImage.at<uchar>(i - 1, j + 1) == 0 && outputImage.at<uchar>(i, j + 1) == 255) b++;
                        if (outputImage.at<uchar>(i, j + 1) == 0 && outputImage.at<uchar>(i + 1, j + 1) == 255) b++;
                        if (outputImage.at<uchar>(i + 1, j + 1) == 0 && outputImage.at<uchar>(i + 1, j) == 255) b++;
                        if (outputImage.at<uchar>(i + 1, j) == 0 && outputImage.at<uchar>(i + 1, j - 1) == 255) b++;
                        if (outputImage.at<uchar>(i + 1, j - 1) == 0 && outputImage.at<uchar>(i, j - 1) == 255) b++;
                        if (outputImage.at<uchar>(i, j - 1) == 0 && outputImage.at<uchar>(i - 1, j - 1) == 255) b++;
                        if (outputImage.at<uchar>(i - 1, j - 1) == 0 && outputImage.at<uchar>(i - 1, j) == 255) b++;
                        if (b == 1) {
                            outputImage.at<uchar>(i, j) = 0;
                            hasChanged = true;
                        }
                    }
                }
            }
        }
    }

    return outputImage;
}

//展示运动轨迹
void show_motion_trajectory(vector<Mat> frames, vector<Point2f> trajectory,int count) {
    // 创建一个空白图像，大小与第一帧图像相同
    Mat result = frames[0].clone();

    // 遍历每一帧图像
    for (int i = 0; i < frames.size(); i++) {
        // 在当前帧图像上绘制运动轨迹
        circle(frames[i], trajectory[i], 3, Scalar(0, 0, 255), -1);

        // 将当前帧图像叠加到结果图像上
        addWeighted(result, 0.5, frames[i], 0.5, 0, result);

        // 保存结果图像
        string filename= "Motion_predicted_" + to_string(count) + ".jpg";
        imwrite(filename, result);

        // 等待一段时间，以便观察
        waitKey(50);
    }
}

// 预测运动物体的移动速度
void predictMotionSpeed(vector<Mat>& frames) {
    // 获取第一帧图像
    Mat prevFrame = frames[0];

    // 定义运动物体的中心点
    Point2f prevCenter, currCenter;

    // 定义运动物体的速度
    float speedX = 0.0f, speedY = 0.0f, speed = 0.0f;

    // 遍历多帧图像
    for (int i = 1; i < frames.size(); i++) {
        // 获取当前帧图像
        Mat currFrame = frames[i];

        // 使用光流法计算运动物体的中心点
        vector<Point2f> prevPoints, currPoints;
        prevPoints.push_back(prevCenter);
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(prevFrame, currFrame, prevPoints, currPoints, status, err);
        currCenter = currPoints[0];

        // 计算运动物体的速度
        speedX = (currCenter.x - prevCenter.x) / (i - 1);
        speedY = (currCenter.y - prevCenter.y) / (i - 1);
        speed = sqrt(speedX * speedX + speedY * speedY);

        // 更新中心点
        prevCenter = currCenter;
        prevFrame = currFrame;

        // 输出速度信息
        cout << "Frame " << i << ": SpeedX = " << speedX << ", SpeedY = " << speedY <<",speed="<<speed<< endl;
    }
}

// 预测多帧图像中运动物体的大小、形状等属性信息
double getContourArea(Mat& img, int threshold,int count)
{
    // Convert the input image to grayscale
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Apply Canny edge detection
    Mat edges;
    Canny(gray, edges, threshold, threshold * 2);
    string filename = "Edge_" + to_string(count) + ".jpg";
    imwrite(filename, edges);

    // Find contours in the edge image
    std::vector<std::vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Calculate the area of the bounding box of the contours
    double area = 0;
    for (auto& contour : contours) {
        Rect bbox = boundingRect(contour);
        area += bbox.area();
    }

    return area;
}