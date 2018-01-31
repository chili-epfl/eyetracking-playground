


#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <QFile>
#include <QTextStream>
#include <QVector>
#include <QDebug>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
struct gaze_info{
    long frame;
    double x,y;
};

bool debug=true;

/*
The goal is to convert the gaze positions in the video to gaze positions in a target image (stimulus).
Sicne the target is planar, we'll use features extraction and homography.

The input video is at video_filename. The person is look at a poster.
The raw gaze event are in the filename_input. This was exported from SMI Begaze.

The stimulus image (in our case the poster) is load in modelImg.
The AOIs are defiend by colors. Each color codes an AOI. The AOI image is in modelAOI.

The gaze positions are stored in filename_output

You need to compile opencv 3 with the contrib module xfeatures.

*/


QString filename_output="/home/chili/lorenzo_demo_output.csv";
QString filename_input="/home/chili/lorenzo_demo_input.txt";
QString video_filename="/home/chili/p1.avi";


int main(int argc, char *argv[])
{
    //Open model image
    cv::Mat modelImg=cv::imread("/home/chili/poster_reduced.JPG");
    //Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    SURF::create();
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> keypoints_modelImg;
    Mat descriptors_modelImg;
    detector->detectAndCompute(modelImg, Mat(), keypoints_modelImg, descriptors_modelImg );

    //Open AOI img. Colors code a specific AOI
    cv::Mat modelAOI;
    modelAOI=cv::imread("/home/chili/poster_AOI.BMP");

    //Read fixations and store them in fixation_list. filename_input is comma separeted
    //PS. The file contains also saccades and other events
    QVector<gaze_info> fixation_list;

    QFile file;
    file.setFileName(filename_input);
    file.open(QFile::ReadOnly);
    QTextStream stream;
    stream.setDevice(&file);
    //Remove header
    stream.readLine();
    while(!stream.atEnd()){
        QString line=stream.readLine();
        QStringList parts=line.split(",");

        gaze_info fixation;
        //Gaze position in the video are at position 3 and 4
        fixation.x=parts[3].toDouble();
        fixation.y=parts[4].toDouble();

        //Video timestamp. It's in the format hh::mm::ss::frame. Frame goes from 0 to 23 (24FPS)
        parts=parts[5].split(":");

        fixation.frame=parts[1].toInt()*60*24+parts[2].toInt()*24+parts[3].toInt()+1;
        fixation_list.append(fixation);
    }
    file.close();

    //Open video
    cv::VideoCapture cap(
                video_filename.toStdString()
                );

    if(!cap.isOpened())  // check if we succeeded
        return -1;
    cv::Mat input_frame;

    //Counters
    int cur_frame=0;
    int current_fix=0;

    //Create a vector of strings for each event in fixations list. We'll store the AOI and the gaze position in the template image

    QVector<QString> AOI_list;
    AOI_list.fill("",fixation_list.size());

    for( ; ; )
    {
        //Read video frame
        cap >> input_frame;
        if(input_frame.empty())
            break;

        qDebug()<<cur_frame;

        //Increase frame counter
        cur_frame++;

        //Stop when all fixations have been analysed
        if(current_fix>=fixation_list.size())
            break;

        //Skip frames without events
        if( cur_frame<fixation_list.at(current_fix).frame)
            continue;

        //Get keypoints in the frame
        std::vector<KeyPoint> keypoints_frame;
        Mat  descriptors_frame;
        detector->detectAndCompute( input_frame, Mat(), keypoints_frame, descriptors_frame );

        FlannBasedMatcher matcher;
        std::vector< DMatch > matches;
        matcher.match( descriptors_modelImg, descriptors_frame, matches );
        double max_dist = 0; double min_dist = 100;
        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_modelImg.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        std::vector< int > good_matches_modelImg, good_matches_frame;
        for( int i = 0; i < descriptors_modelImg.rows; i++ )
        { if( matches[i].distance <= max(2*min_dist, 0.02) )
            { good_matches_modelImg.push_back( matches[i].queryIdx);
                good_matches_frame.push_back( matches[i].trainIdx);}
        }

        //If we have enough correspondences
        if(good_matches_modelImg.size()>10){

            std::vector<Point2f> src,dst;
            KeyPoint::convert(keypoints_frame,src,good_matches_frame);
            KeyPoint::convert(keypoints_modelImg,dst,good_matches_modelImg);


            //Map gaze on target image
            auto H= cv::findHomography(src,dst,cv::RANSAC);
            cv::Mat gaze= cv::Mat(3, 1, CV_64F);
            gaze.at<double>(0)=fixation_list.at(current_fix).x;
            gaze.at<double>(1)=fixation_list.at(current_fix).y;
            gaze.at<double>(2)=1;

            //This is the gaze projected in the template image
            cv::Mat gaze_P=H*gaze;

            int x=gaze_P.at<double>(0)/gaze_P.at<double>(2);
            int y=gaze_P.at<double>(1)/gaze_P.at<double>(2);

            //Get color in AOI image
            int b=0,g=0,r=0;

            if(x>=0 && x<modelAOI.cols && y>=0 && y<modelAOI.rows){
                b=modelAOI.at<cv::Vec3b>(y,x)[0];
                g=modelAOI.at<cv::Vec3b>(y,x)[1];
                r=modelAOI.at<cv::Vec3b>(y,x)[2];
            }
            //Get the AOI
            QString AOI;
            switch (r) {
            case 5:
                AOI="Header";
                break;
            case 10:
                AOI="Robot";
                break;
            case 15:
                AOI="Setup";
                break;
            case 20:
                AOI="Results";
                break;
            case 25:
                AOI="Project";
                break;
            case 30:
                AOI="Footer";
                break;
            case 35:
                AOI="Exp_States";
                break;

            }

            //Store AOI and gaze position
            AOI=AOI.append(",");
            AOI=AOI.append(QString::number(x));
            AOI=AOI.append(",");
            AOI=AOI.append(QString::number(y));
            AOI_list.replace(current_fix,AOI);



            //*******
            if(debug){
                AOI=QString("Gaze:")+AOI;
                cv::Point2f center(fixation_list.at(current_fix).x,fixation_list.at(current_fix).y);
                cv::circle(input_frame,center,10,cvScalar(0,0,255),5);
                cv::putText(input_frame, AOI.toStdString(), center,
                            cv::FONT_HERSHEY_SIMPLEX, 0.5f, cvScalar(255));
                cv::imshow("VideoFrame", input_frame);

                center.x=x;
                center.y=y;
                cv::Mat current_s_AOI_clone=modelImg.clone();
                cv::circle(current_s_AOI_clone,center,10,cvScalar(0,0,255),5);
                cv::putText(current_s_AOI_clone, AOI.toStdString(), center,
                            cv::FONT_HERSHEY_SIMPLEX, 1, cvScalar(255));
                cv::imshow("RefrenceFrame", current_s_AOI_clone);
                cv::waitKey(20); // waits to display frame

            }
            //*******
        }
        current_fix++;
    }

    //Create output file
    file.open(QFile::ReadOnly);
    stream.setDevice(&file);

    QFile output_file;
    output_file.setFileName(filename_output);
    output_file.open(QFile::WriteOnly);

    QTextStream outputstream;
    outputstream.setDevice(&output_file);

    int counter=0;
    //header
    outputstream<<stream.readLine()<<","<<"AOI"<<","<<"x_ref"<<","<<"y_ref"<<"\n";
    while(!stream.atEnd()){
        QString line=stream.readLine();
        outputstream<<line<<","<<AOI_list[counter]<<"\n";
        counter++;
    }
    file.close();
    outputstream.flush();
    output_file.close();

    return 0;
}
