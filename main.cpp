#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include "func.h " 

using namespace cv;
using namespace std;


int main()
{
    // 定义变量
    Mat frame, gray, pre_frame;
    Mat seg, glcm_image;
    int count = 0;
    time_t start_time, end_time;
    vector<Mat> frames;
    vector<Mat> segs;

    // 打开摄像机
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cout << "无法打开摄像机" << endl;
        return -1;
    }

    // 获取当前时间
    time(&start_time);

    while (true)
    {
        // 读取当前帧
        cap >> frame;

        // 转换为灰度图像
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 如果是第一帧，则将其作为前一帧
        if (pre_frame.empty())
        {
            gray.copyTo(pre_frame);
            continue;
        }

        // 计算当前帧和前一帧的差异
        Mat diff;
        absdiff(pre_frame, gray, diff);

        // 二值化差异图像
        threshold(diff, diff, 30, 255, THRESH_BINARY);

        // 计算差异图像中非零像素的个数
        int non_zero_count = countNonZero(diff);

        // 如果非零像素的个数超过一定阈值，则保存当前帧
        if (non_zero_count > 1000)
        {
            // 获取当前时间
            time(&end_time);

            // 如果时间间隔大于1秒，则保存当前帧
            if (difftime(end_time, start_time) > 1)
            {
                printf("注意，检测到物体入侵！\n这是保存的第%d批入侵物体相关的图像\n",count+1);
                // 定义图像保存名称
                string filename_0 = "Image_" + to_string(count) + ".jpg";
                string filename_1 = "BluredImage_" + to_string(count) + ".jpg";
                string filename_2 = "SegedImage_" + to_string(count) + ".jpg";
                string filename_3 = "SkeedImage_" + to_string(count) + ".jpg";
                string filename_4 = "Matrix_" + to_string(count) + ".jpg";

                // 保存当前帧
                imwrite(filename_0, frame);
                cout << "Saved image: " << filename_0 << endl;

                // 高斯平滑
                Mat gaussianBlured = gaussianBlur(frame, 5, 1.5);
                imwrite(filename_1, gaussianBlured);
                cout << "Saved image: " << filename_1 << endl;

                // 灰度分割
                Mat graySeged;
                int thr = getThresholdValue(gray);
                grayScaleSegmentation(gray, graySeged, thr);
                imwrite(filename_2, graySeged);
                cout << "Saved image: " << filename_2 << endl;

                // 图像细化
                imwrite(filename_3, stentifordThinning(graySeged));
                cout << "Saved image: " << filename_3 << endl;

                // 计算灰度共生矩阵和对比度
                Mat coOccurrenceMatrix = grayLevelCoOccurrenceMatrix(frame, 1, 0, 256);
                float contrastValue = contrast(coOccurrenceMatrix);
                normalize(coOccurrenceMatrix, glcm_image, 0, 255, NORM_MINMAX, CV_8UC1);
                convertScaleAbs(glcm_image, glcm_image);
                imshow("coOccurrenceMatrix", glcm_image);
                imwrite(filename_4, glcm_image);
                cout << "Contrast ofc o-occurrence matrix: " << contrastValue << endl;

                // 将各帧图像用向量存储
                Mat temp = imread(filename_0);
                Mat temp_0 = imread(filename_2);
                frames.push_back(temp);
                segs.push_back(temp_0);

                // 记录运动属性
                double area = getContourArea(frame, thr, count);
                cout << "Area of invasion: " << area << endl;

                // 更新计数器和时间
                count++;
                time(&start_time);
            }
        }

        // 显示当前帧
        imshow("frame", frame);
        
        // 显示当前帧经过灰度分割后的图像
        int thr = getThresholdValue(gray);
        grayScaleSegmentation(gray, seg, thr);
        imshow("seg", seg);

        // 等待按键
        if (waitKey(30) == 27)
        {
            break;
        }

        // 将当前帧作为前一帧
        gray.copyTo(pre_frame);
    }

    // 将各帧图片前景重叠以展示前景的轨迹
    int count_0 = 0;
    int count_1 = 1;
    vector<Point2f> trajectory;
    trajectory.push_back(Point2f(100, 100));
    trajectory.push_back(Point2f(120, 120));
    trajectory.push_back(Point2f(140, 140));
    trajectory.push_back(Point2f(160, 160));
    show_motion_trajectory(frames, trajectory,count_0);
    show_motion_trajectory(segs, trajectory,count_1);

    // 预测运动物体的轨迹
    predictMotionSpeed(frames);

    // 释放摄像机
    cap.release();

    return 0;
}