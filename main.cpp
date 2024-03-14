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
    // �������
    Mat frame, gray, pre_frame;
    Mat seg, glcm_image;
    int count = 0;
    time_t start_time, end_time;
    vector<Mat> frames;
    vector<Mat> segs;

    // �������
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cout << "�޷��������" << endl;
        return -1;
    }

    // ��ȡ��ǰʱ��
    time(&start_time);

    while (true)
    {
        // ��ȡ��ǰ֡
        cap >> frame;

        // ת��Ϊ�Ҷ�ͼ��
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // ����ǵ�һ֡��������Ϊǰһ֡
        if (pre_frame.empty())
        {
            gray.copyTo(pre_frame);
            continue;
        }

        // ���㵱ǰ֡��ǰһ֡�Ĳ���
        Mat diff;
        absdiff(pre_frame, gray, diff);

        // ��ֵ������ͼ��
        threshold(diff, diff, 30, 255, THRESH_BINARY);

        // �������ͼ���з������صĸ���
        int non_zero_count = countNonZero(diff);

        // ����������صĸ�������һ����ֵ���򱣴浱ǰ֡
        if (non_zero_count > 1000)
        {
            // ��ȡ��ǰʱ��
            time(&end_time);

            // ���ʱ��������1�룬�򱣴浱ǰ֡
            if (difftime(end_time, start_time) > 1)
            {
                printf("ע�⣬��⵽�������֣�\n���Ǳ���ĵ�%d������������ص�ͼ��\n",count+1);
                // ����ͼ�񱣴�����
                string filename_0 = "Image_" + to_string(count) + ".jpg";
                string filename_1 = "BluredImage_" + to_string(count) + ".jpg";
                string filename_2 = "SegedImage_" + to_string(count) + ".jpg";
                string filename_3 = "SkeedImage_" + to_string(count) + ".jpg";
                string filename_4 = "Matrix_" + to_string(count) + ".jpg";

                // ���浱ǰ֡
                imwrite(filename_0, frame);
                cout << "Saved image: " << filename_0 << endl;

                // ��˹ƽ��
                Mat gaussianBlured = gaussianBlur(frame, 5, 1.5);
                imwrite(filename_1, gaussianBlured);
                cout << "Saved image: " << filename_1 << endl;

                // �Ҷȷָ�
                Mat graySeged;
                int thr = getThresholdValue(gray);
                grayScaleSegmentation(gray, graySeged, thr);
                imwrite(filename_2, graySeged);
                cout << "Saved image: " << filename_2 << endl;

                // ͼ��ϸ��
                imwrite(filename_3, stentifordThinning(graySeged));
                cout << "Saved image: " << filename_3 << endl;

                // ����Ҷȹ�������ͶԱȶ�
                Mat coOccurrenceMatrix = grayLevelCoOccurrenceMatrix(frame, 1, 0, 256);
                float contrastValue = contrast(coOccurrenceMatrix);
                normalize(coOccurrenceMatrix, glcm_image, 0, 255, NORM_MINMAX, CV_8UC1);
                convertScaleAbs(glcm_image, glcm_image);
                imshow("coOccurrenceMatrix", glcm_image);
                imwrite(filename_4, glcm_image);
                cout << "Contrast ofc o-occurrence matrix: " << contrastValue << endl;

                // ����֡ͼ���������洢
                Mat temp = imread(filename_0);
                Mat temp_0 = imread(filename_2);
                frames.push_back(temp);
                segs.push_back(temp_0);

                // ��¼�˶�����
                double area = getContourArea(frame, thr, count);
                cout << "Area of invasion: " << area << endl;

                // ���¼�������ʱ��
                count++;
                time(&start_time);
            }
        }

        // ��ʾ��ǰ֡
        imshow("frame", frame);
        
        // ��ʾ��ǰ֡�����Ҷȷָ���ͼ��
        int thr = getThresholdValue(gray);
        grayScaleSegmentation(gray, seg, thr);
        imshow("seg", seg);

        // �ȴ�����
        if (waitKey(30) == 27)
        {
            break;
        }

        // ����ǰ֡��Ϊǰһ֡
        gray.copyTo(pre_frame);
    }

    // ����֡ͼƬǰ���ص���չʾǰ���Ĺ켣
    int count_0 = 0;
    int count_1 = 1;
    vector<Point2f> trajectory;
    trajectory.push_back(Point2f(100, 100));
    trajectory.push_back(Point2f(120, 120));
    trajectory.push_back(Point2f(140, 140));
    trajectory.push_back(Point2f(160, 160));
    show_motion_trajectory(frames, trajectory,count_0);
    show_motion_trajectory(segs, trajectory,count_1);

    // Ԥ���˶�����Ĺ켣
    predictMotionSpeed(frames);

    // �ͷ������
    cap.release();

    return 0;
}