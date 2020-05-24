// // CPP program to detects face in a video 
  
// // Include required header files from OpenCV directory 
// #include <opencv4/opencv2/objdetect.hpp> 
// #include <opencv4/opencv2/highgui.hpp>
// #include <opencv4/opencv2/imgproc.hpp>
// #include <iostream> 
  
// using namespace std; 
// using namespace cv; 
  
// // Function for Face Detection 
// void detectAndDraw( Mat& img, CascadeClassifier& cascade,  
//                 CascadeClassifier& nestedCascade, double scale );
// void drawCircle(Rect& r);
// string cascadeName, nestedCascadeName; 
  
// int main( int argc, const char** argv ) 
// { 
//     // VideoCapture class for playing video for which faces to be detected 
//     VideoCapture capture;  
//     Mat frame, image; 
  
//     // PreDefined trained XML classifiers with facial features 
//     CascadeClassifier cascade, nestedCascade;  
//     double scale=1; 
  
//     // Load classifiers from "opencv/data/haarcascades" directory  
//     nestedCascade.load( "../../opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml" ) ; 
  
//     // Change path before execution  
//     cascade.load( "../../opencv/data/haarcascades/haarcascade_frontalcatface.xml" ) ;  
  
//     // Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video 
//     capture.open(2);  
//     if( capture.isOpened() ) 
//     { 
//         // Capture frames from video and detect faces 
//         cout << "Face Detection Started...." << endl; 
//         while(1) 
//         { 
//             capture >> frame; 
//             if( frame.empty() ) 
//                 break; 
//             Mat frame1 = frame.clone(); 
//             detectAndDraw( frame1, cascade, nestedCascade, scale );  
//             char c = (char)waitKey(10); 
          
//             // Press q to exit from window 
//             if( c == 27 || c == 'q' || c == 'Q' )  
//                 break; 
//         } 
//     } 
//     else
//         cout<<"Could not Open Camera"; 
//     return 0; 
// } 
  
// void detectAndDraw( Mat& img, CascadeClassifier& cascade, 
//                     CascadeClassifier& nestedCascade, 
//                     double scale) 
// { 
//     vector<Rect> faces, faces2; 
//     Mat gray, smallImg; 
  
//     cvtColor( img, gray, COLOR_BGR2GRAY ); // Convert to Gray Scale 
//     double fx = 1 / scale; 
  
//     // Resize the Grayscale Image  
//     resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );  
//     equalizeHist( smallImg, smallImg ); 
  
//     // Detect faces of different sizes using cascade classifier  
//     cascade.detectMultiScale( smallImg, faces, 1.1,  
//                             2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) ); 
  
//     // Draw circles around the faces 
//     for ( size_t i = 0; i < faces.size(); i++ ) 
//     { 
//         Rect r = faces[i]; 
//         Mat smallImgROI; 
//         vector<Rect> nestedObjects; 
//         Point center; 
//         Scalar color = Scalar(255, 0, 0); // Color for Drawing tool 
//         int radius; 
  
//         double aspect_ratio = static_cast<double>(r.width) / static_cast<double>(r.height); 
//         if( 0.75 < aspect_ratio && aspect_ratio < 1.3 ) 
//         { 
//             center.x = cvRound((r.x + r.width / 2) * scale); 
//             center.y = cvRound((r.y + r.height / 2) * scale); 
//             radius = cvRound((r.width + r.height) / 4 * scale); 
//             circle( img, center, radius, color, 3, 8, 0 ); 
//         } 
//         else
//             rectangle( img, cv::Point(cvRound(r.x * scale), cvRound(r.y * scale)), 
//                     cv::Point(cvRound((r.x + r.width - 1) * scale),  
//                     cvRound((r.y + r.height - 1) * scale)), color, 3, 8, 0); 
//         if( nestedCascade.empty() ) 
//             continue; 
//         smallImgROI = smallImg( r ); 
          
//         // Detection of eyes int the input image 
//         nestedCascade.detectMultiScale( smallImgROI, nestedObjects, 1.1, 2, 
//                                         0|CASCADE_SCALE_IMAGE, Size(30, 30) );  
          
//         // Draw circles around eyes 
//         for ( size_t j = 0; j < nestedObjects.size(); j++ )  
//         { 
//             Rect nr = nestedObjects[j]; 
//             center.x = cvRound((r.x + nr.x + nr.width * 0.5) * scale); 
//             center.y = cvRound((r.y + nr.y + nr.height * 0.5) * scale); 
//             radius = cvRound((nr.width + nr.height) * 0.25 * scale); 
//             circle( img, center, radius, color, 3, 8, 0 ); 
//         } 
//     } 
  
//     // Show Processed Image with detected faces 
//     imshow( "Face Detection", img );  
// }

// void drawCircle(Rect& r)
// {
//     // center.x = cvRound((r.x + r.width / 2) * scale); 
//     // center.y = cvRound((r.y + r.height / 2) * scale); 
//     // radius = cvRound((r.width + r.height) / 4 * scale); 
//     // circle( img, center, radius, color, 3, 8, 0 ); 
// }
 

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv4/opencv2/contrib.hpp>
// #include <opencv4/opencv2/opencv.hpp>
// #include <opencv4/opencv2/>
#include <iostream>
#include <thread>

using namespace std;
using namespace cv;
using namespace cv::dnn;

#define M_PI  3.14159265358979323846
#define EPS  1e-5


struct FaceBox
{
    int x, y, w, h;
    float angle, scale, score;

    Rect getBox() { return Rect(x,y,w,h); }

    FaceBox(int x_, int y_, int w_, int h_, float a_, float s_, float c_)
    : x(x_), y(y_), w(w_), h(h_), angle(a_), scale(s_), score(c_)
    {}
};


bool xyValid(int _x, int _y, cv::Mat _img)
{
    if (_x >= 0 && _x < _img.cols && _y >= 0 && _y < _img.rows)
        return true;
    else
        return false;
}


cv::Mat preprocessImg(const cv::Mat& _img)
{
    cv::Mat mean(_img.size(), CV_32FC3, cv::Scalar(104, 117, 123));
    cv::Mat imgF;
    _img.convertTo(imgF, CV_32FC3);
    return imgF - mean;
}

cv::Mat preprocessImg(const cv::Mat& _img, int _dim)
{
    cv::Mat imgNew;
    cv::resize(_img, imgNew, cv::Size(_dim, _dim));
    cv::Mat mean(imgNew.size(), CV_32FC3, cv::Scalar(104, 117, 123));
    cv::Mat imgF;
    imgNew.convertTo(imgF, CV_32FC3);
    return imgF - mean;
}


cv::Mat resizeImg(cv::Mat& _img, float _scale)
{
    cv::Mat aux;
    cv::resize(_img, aux, cv::Size(int(_img.cols / _scale), int(_img.rows / _scale)));
    return aux;
}

cv::Mat padImg(cv::Mat _img)
{
    int row = std::min(int(_img.rows * 0.2), 100);
    int col = std::min(int(_img.cols * 0.2), 100);
    cv::Mat aux;
    cv::copyMakeBorder(_img, aux, row, row, col, col, cv::BORDER_CONSTANT);
    return aux;
}


bool compareFaceByScore(const FaceBox &box1, const FaceBox &box2)
{
    return box1.score > box2.score;
}


float IoU(FaceBox &box1, FaceBox &box2)
{
    int xOverlap = std::max(0, std::min(box1.x + box1.w - 1, box2.x + box2.w - 1) - std::max(box1.x, box2.x) + 1);
    int yOverlap = std::max(0, std::min(box1.y + box1.h - 1, box2.y + box2.h - 1) - std::max(box1.y, box2.y) + 1);
    int intersection = xOverlap * yOverlap;
    int unio = box1.w * box1.h + box2.w * box2.h - intersection;

    return float(intersection) / unio;
}


std::vector<FaceBox> NMS(std::vector<FaceBox> &_faces, bool _local, float _threshold)
{
    if (_faces.size() == 0)
        return _faces;

    std::sort(_faces.begin(), _faces.end(), compareFaceByScore);
    vector<int> flag(_faces.size(), 0);
    for (int i = 0; i < _faces.size(); i++)
    {
        if (flag[i])
            continue;
        for (int j = i + 1; j < _faces.size(); j++)
        {
            if (_local && abs(_faces[i].scale - _faces[j].scale) > EPS)
                continue;

            if (IoU(_faces[i], _faces[j]) > _threshold)
                flag[j] = 1;
        }
    }

    std::vector<FaceBox> faces_nms;
    for (int i = 0; i < _faces.size(); i++)
    {
        if (!flag[i])
            faces_nms.push_back(_faces[i]);
    }
    return faces_nms;
}


std::vector<FaceBox> TransformBoxes(cv::Mat _img, cv::Mat _imgPad, std::vector<FaceBox> &_faces)
{
    int row = (_imgPad.rows - _img.rows) / 2;
    int col = (_imgPad.cols - _img.cols) / 2;

    std::vector<FaceBox> newFaceBoxes;
    for(int i = 0; i < _faces.size(); i++)
    {
        if (_faces[i].w > 0 && _faces[i].h > 0)
            newFaceBoxes.push_back(FaceBox(_faces[i].x - col, _faces[i].y - row, _faces[i].w, _faces[i].w, _faces[i].angle, _faces[i].scale, _faces[i].score));
    }
    return newFaceBoxes;
}



std::vector<FaceBox> PCN_1(cv::Mat _img, cv::Mat _paddedImg, cv::dnn::Net _net, float _thresh, int _minFaceSize)
{
    std::vector<FaceBox> faceBoxes_1;
    int row = (_paddedImg.rows - _img.rows) / 2;
    int col = (_paddedImg.cols - _img.cols) / 2;
    int netSize = 24;
    int minFace = _minFaceSize * 1.4; // - size 20 + 40%
    float currentScale = minFace / float(netSize);
    int stride = 8;

    cv::Mat resizedImg = resizeImg(_img, currentScale);

    while (std::min(resizedImg.rows, resizedImg.cols) >= netSize)
    {
        // - Set input for net
        Mat inputMat = preprocessImg(resizedImg);
        Mat inputBlob = blobFromImage(inputMat, 1.0, Size(), Scalar(), false, false);
        _net.setInput(inputBlob);

        std::vector<String> outputBlobNames = {"cls_prob", "rotate_cls_prob", "bbox_reg_1" };
        std::vector<cv::Mat> outputBlobs;

        _net.forward(outputBlobs, outputBlobNames);
        cv::Mat scoresData = outputBlobs[0];
        cv::Mat rotateProbsData = outputBlobs[1];
        cv::Mat regressionData = outputBlobs[2];

        // scoresData.ptr<float>(0, 1)  ---->  image 0, channel 1
        Mat scoresMat(scoresData.size[2], scoresData.size[3], CV_32F, scoresData.ptr<float>(0, 1));
        Mat reg_1_Mat(regressionData.size[2], regressionData.size[3], CV_32F, regressionData.ptr<float>(0, 0));
        Mat reg_2_Mat(regressionData.size[2], regressionData.size[3], CV_32F, regressionData.ptr<float>(0, 1));
        Mat reg_3_Mat(regressionData.size[2], regressionData.size[3], CV_32F, regressionData.ptr<float>(0, 2));
        Mat rotateProbsMat(rotateProbsData.size[2], rotateProbsData.size[3], CV_32F, rotateProbsData.ptr<float>(0, 1));

        float w = netSize * currentScale;

        for (int i = 0; i < scoresData.size[2]; i++)
        {
            for (int j = 0; j < scoresData.size[3]; j++)
            {
                if (scoresMat.at<float>(i, j) < _thresh)
                    {continue;}

                float score = scoresMat.at<float>(i, j);
                float sn = reg_1_Mat.at<float>(i, j);
                float xn = reg_2_Mat.at<float>(i, j);
                float yn = reg_3_Mat.at<float>(i, j);

                int x = int(j * currentScale * stride - 0.5 * sn * w + sn * xn * w + 0.5 * w) + col;
                int y = int(i * currentScale * stride - 0.5 * sn * w + sn * yn * w + 0.5 * w) + row;
                int wh = int(w * sn);

                if(xyValid(x, y, _paddedImg) && xyValid(x+wh-1, y+wh-1, _paddedImg))
                {
                    if (rotateProbsMat.at<float>(i, j) > 0.5)
                    {
                        faceBoxes_1.push_back(FaceBox(x,y,wh,wh,0, currentScale, score));
                        //rectangle(_paddedImg, Rect(x, y, wh, wh), Scalar(0, 255, 0), 2);
                    }
                    else
                    {
                        faceBoxes_1.push_back(FaceBox(x,y,wh,wh,180, currentScale, score));
                        //rectangle(_paddedImg, Rect(x, y, wh, wh), Scalar(255, 0, 0), 2);
                    }
                }


            }
        }

        resizedImg = resizeImg(resizedImg, 1.414);
        currentScale = float(_img.rows) / resizedImg.rows;
    }

    return faceBoxes_1;
}


std::vector<FaceBox> PCN_2(cv::Mat _img, cv::Mat _img180, cv::dnn::Net _net, float _threshold, int _dim, std::vector<FaceBox> _faces)
{
    //_dim = 24 ---> network size
    if (_faces.size() == 0)
        return _faces;

    std::vector<cv::Mat> dataList;
    int height = _img.rows;
    for (int i = 0; i < _faces.size(); i++)
    {
        if (abs(_faces[i].angle) < EPS)
            dataList.push_back(preprocessImg(_img(_faces[i].getBox()), _dim));
        else
        {
            int y2 = _faces[i].y + _faces[i].h - 1;
            dataList.push_back(preprocessImg(_img180(cv::Rect(_faces[i].x, height - 1 - y2, _faces[i].w, _faces[i].h)), _dim));
        }
    }

    std::vector<FaceBox> faceBoxes_2;

    // - Process the dataList vector
    for(int dataNr=0; dataNr<dataList.size(); dataNr++)
    {
        Mat inputBlob = blobFromImage(dataList[dataNr], 1.0, Size(), Scalar(), false, false);
        _net.setInput(inputBlob);
        std::vector<String> outputBlobNames = {"cls_prob", "rotate_cls_prob", "bbox_reg_2" };
        std::vector<cv::Mat> outputBlobs;

        _net.forward(outputBlobs, outputBlobNames);
        cv::Mat scoresData = outputBlobs[0];
        cv::Mat rotateProbsData = outputBlobs[1];
        cv::Mat regressionData = outputBlobs[2];

        Mat scoresMat(scoresData.size[1], scoresData.size[0], CV_32F, scoresData.ptr<float>(0, 0)); // - using channel 0
        Mat regMat(regressionData.size[1], regressionData.size[0], CV_32F, regressionData.ptr<float>(0, 0));
        Mat rotateProbsMat(rotateProbsData.size[1], rotateProbsData.size[0], CV_32F, rotateProbsData.ptr<float>(0, 0));

        float score = scoresMat.at<float>(1,0);
        if(score < _threshold)
            continue;

        float sn = regMat.at<float>(0,0);// reg->data_at(i, 0, 0, 0);
        float xn = regMat.at<float>(1,0);// reg->data_at(i, 1, 0, 0);
        float yn = regMat.at<float>(2,0);//reg->data_at(i, 2, 0, 0);


        int cropX = _faces[dataNr].x;
        int cropY = _faces[dataNr].y;
        int cropW = _faces[dataNr].w;
        if (abs(_faces[dataNr].angle)  > EPS)
            cropY = height - 1 - (cropY + cropW - 1);
        int w = int(sn * cropW);
        int x = int(cropX  - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW);
        int y = int(cropY  - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW);
        float maxRotateScore = 0;
        int maxRotateIndex = 0;
        for (int j = 0; j < 3; j++)
        {
            float rotateProb = rotateProbsMat.at<float>(j,0);
            if (rotateProb > maxRotateScore)
            {
                maxRotateScore = rotateProb;//rotateProb->data_at(i, j, 0, 0);
                maxRotateIndex = j;
            }
        }

        if(xyValid(x, y, _img) && xyValid(x+w-1, y+w-1, _img)) //(Legal(x, y, img) && Legal(x + w - 1, y + w - 1, img))
        {
            float angle = 0;
            if (abs(_faces[dataNr].angle)  < EPS)
            {
                if (maxRotateIndex == 0)
                    angle = 90;
                else if (maxRotateIndex == 1)
                    angle = 0;
                else
                    angle = -90;
                faceBoxes_2.push_back(FaceBox(x,y,w,w,angle, _faces[dataNr].scale, score));
                //ret.push_back(Window2(x, y, w, w, angle, _faces[i].scale, prob->data_at(i, 1, 0, 0)));
            }
            else
            {
                if (maxRotateIndex == 0)
                    angle = 90;
                else if (maxRotateIndex == 1)
                    angle = 180;
                else
                    angle = -90;
                faceBoxes_2.push_back(FaceBox(x, height - 1 -  (y + w - 1), w, w, angle, _faces[dataNr].scale, score));
            }
        }
    }
    return faceBoxes_2;
}



std::vector<FaceBox> PCN_3(cv::Mat _img, cv::Mat _img180, cv::Mat _img90, cv::Mat _imgNeg90, cv::dnn::Net _net, float _threshold, int _dim, std::vector<FaceBox> _faces)
{
    if (_faces.size() == 0)
        return _faces;

    std::vector<cv::Mat> dataList;
    int height = _img.rows;
    int width = _img.cols;
    for (int i = 0; i < _faces.size(); i++)
    {
        if (abs(_faces[i].angle) < EPS)
            dataList.push_back(preprocessImg(_img(cv::Rect(_faces[i].x, _faces[i].y, _faces[i].w, _faces[i].h)), _dim));
        else if (abs(_faces[i].angle - 90) < EPS)
        {
            dataList.push_back(preprocessImg(_img90(cv::Rect(_faces[i].y, _faces[i].x, _faces[i].h, _faces[i].w)), _dim));
        }
        else if (abs(_faces[i].angle + 90) < EPS)
        {
            int x = _faces[i].y;
            int y = width - 1 - (_faces[i].x + _faces[i].w - 1);
            dataList.push_back(preprocessImg(_imgNeg90(cv::Rect(x, y, _faces[i].w, _faces[i].h)), _dim));
        }
        else
        {
            int y2 = _faces[i].y + _faces[i].h - 1;
            dataList.push_back(preprocessImg(_img180(cv::Rect(_faces[i].x, height - 1 - y2, _faces[i].w, _faces[i].h)), _dim));
        }
    }

    std::vector<FaceBox> faceBoxes_3;

    for(int dataNr=0; dataNr<dataList.size(); dataNr++)
    {
        Mat inputBlob = blobFromImage(dataList[dataNr], 1.0, Size(), Scalar(), false, false);
        _net.setInput(inputBlob);
        std::vector<String> outputBlobNames = {"cls_prob", "rotate_reg_3", "bbox_reg_3" };
        std::vector<cv::Mat> outputBlobs;

        _net.forward(outputBlobs, outputBlobNames);
        cv::Mat scoresData = outputBlobs[0];
        cv::Mat rotateProbsData = outputBlobs[1];
        cv::Mat regressionData = outputBlobs[2];

        Mat scoresMat(scoresData.size[1], scoresData.size[0], CV_32F, scoresData.ptr<float>(0, 0)); // - using channel 0
        Mat regMat(regressionData.size[1], regressionData.size[0], CV_32F, regressionData.ptr<float>(0, 0));
        Mat rotateProbsMat(rotateProbsData.size[1], rotateProbsData.size[0], CV_32F, rotateProbsData.ptr<float>(0, 0));

        // - score = prob->data_at(i, 1, 0, 0)
        float score = scoresMat.at<float>(1,0);
        if(score < _threshold)
            continue;

        float sn = regMat.at<float>(0,0);// reg->data_at(i, 0, 0, 0);
        float xn = regMat.at<float>(1,0);// reg->data_at(i, 1, 0, 0);
        float yn = regMat.at<float>(2,0);//reg->data_at(i, 2, 0, 0);

        int cropX = _faces[dataNr].x;
        int cropY = _faces[dataNr].y;
        int cropW = _faces[dataNr].w;
        cv::Mat imgTmp = _img;
        if (abs(_faces[dataNr].angle - 180)  < EPS)
        {
            cropY = height - 1 - (cropY + cropW - 1);
            imgTmp = _img180;
        }
        else if (abs(_faces[dataNr].angle - 90)  < EPS)
        {
            std::swap(cropX, cropY);
            imgTmp = _img90;
        }
        else if (abs(_faces[dataNr].angle + 90)  < EPS)
        {
            cropX = _faces[dataNr].y;
            cropY = width - 1 - (_faces[dataNr].x + _faces[dataNr].w - 1);
            imgTmp = _imgNeg90;
        }

        int w = int(sn * cropW);
        int x = int(cropX  - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW);
        int y = int(cropY  - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW);
        float angleRange_ = 45;
        float angle = angleRange_ * rotateProbsMat.at<float>(0,0);
        if(xyValid(x, y, imgTmp) && xyValid(x + w - 1, y + w - 1, imgTmp))
        {
            if (abs(_faces[dataNr].angle)  < EPS)
                faceBoxes_3.push_back(FaceBox(x, y, w, w, angle, _faces[dataNr].scale, score));
            else if (abs(_faces[dataNr].angle - 180)  < EPS)
            {
                faceBoxes_3.push_back(FaceBox(x, height - 1 -  (y + w - 1), w, w, 180 - angle, _faces[dataNr].scale, score));
            }
            else if (abs(_faces[dataNr].angle - 90)  < EPS)
            {
                faceBoxes_3.push_back(FaceBox(y, x, w, w, 90 - angle, _faces[dataNr].scale, score));
            }
            else
            {
                faceBoxes_3.push_back(FaceBox(width - y - w, x, w, w, -90 + angle, _faces[dataNr].scale, score));
            }
        }
    }

    return faceBoxes_3;
}



cv::Point RotatePoint(int x, int y, float centerX, float centerY, float angle)
{
    x -= centerX;
    y -= centerY;
    float theta = -angle * M_PI / 180;
    int rx = int(centerX + x * std::cos(theta) - y * std::sin(theta));
    int ry = int(centerY + x * std::sin(theta) + y * std::cos(theta));
    return cv::Point(rx, ry);
}

void DrawLine(cv::Mat img, std::vector<cv::Point> pointList)
{
    int thick = 2;
    cv::Scalar cyan = CV_RGB(0, 0, 255);
    cv::Scalar blue = CV_RGB(0, 255, 0);
    cv::line(img, pointList[0], pointList[1], cyan, thick);
    cv::line(img, pointList[1], pointList[2], cyan, thick);
    cv::line(img, pointList[2], pointList[3], cyan, thick);
    cv::line(img, pointList[3], pointList[0], blue, thick);
}

void DrawFace(cv::Mat img, FaceBox face)
{
    int x1 = face.x;
    int y1 = face.y;
    int x2 = face.w + face.x - 1;
    int y2 = face.w + face.y - 1;
    int centerX = (x1 + x2) / 2;
    int centerY = (y1 + y2) / 2;
    std::vector<cv::Point> pointList;
    pointList.push_back(RotatePoint(x1, y1, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x1, y2, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x2, y2, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x2, y1, centerX, centerY, face.angle));

                string box_text = format("Prediction = %d", 10);
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face.x - 10, 0);
            int pos_y = std::max(face.y - 10, 0);
            // And now put it into the image:
            putText(img, box_text, RotatePoint(pos_x, pos_y, centerX, centerY, face.angle), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
    DrawLine(img, pointList);
}

vector<Mat> images;
vector<int> labels;
// ------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    labels.resize(2);
    for (int i = 0; i < labels.size(); i++)
        labels[i] = i;
    Net net_1 = readNet("/usr/local/share/pcn/PCN-1.prototxt", "/usr/local/share/pcn/PCN.caffemodel");
    Net net_2 = readNet("/usr/local/share/pcn/PCN-2.prototxt", "/usr/local/share/pcn/PCN.caffemodel");
    Net net_3 = readNet("/usr/local/share/pcn/PCN-3.prototxt", "/usr/local/share/pcn/PCN.caffemodel");

	cv::VideoCapture capture(0);

	cv::Mat img;

	while (!0)
	{
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		//read next frame
		if (!capture.read(img))
		{
			std::cout << "Failed to read video!" << std::endl;
			return -1;
		}
        images.push_back(img);
        // std::this_thread::sleep_for(std::chrono::seconds(10));
        if(images.size() == 2)
        {
            Mat testSample = images[images.size() - 1];

            // images.pop_back();
            // labels.pop_back();

            Ptr<cv::face::FaceRecognizer> model = cv::face::EigenFaceRecognizer::create();//cv::face::createEigenFaceRecognizer();
            model->train(images, labels);
            // The following line predicts the label of a given
            // test image:
            cvtColor(testSample, testSample, COLOR_BGR2GRAY);

            int predictedLabel = model->predict(testSample);
            std::cout << "LABEL: " << predictedLabel << std::endl;

            double confidence = 0.0;
            model->predict(testSample, predictedLabel, confidence);
            std::cout << "CONF: " << confidence << std::endl;
        }


		cv::flip(img, img, 1);//Y-axis mirror (i.e. horizontal mirror)
		
		cv::Mat paddedImg = padImg(img);

		cv::Mat img180, img90, imgNeg90;
		cv::flip(paddedImg, img180, 0);
		cv::transpose(paddedImg, img90);
		cv::flip(img90, imgNeg90, 0);

		float thresholds[] = { 0.37, 0.43, 0.95 };

		cv::TickMeter tm;
		tm.reset();
		tm.start();

		int minFaceSize = 40;
		std::vector<FaceBox> faces = PCN_1(img, paddedImg, net_1, thresholds[0], minFaceSize);
		faces = NMS(faces, true, 0.8);
		faces = PCN_2(paddedImg, img180, net_2, thresholds[1], 24, faces);
		faces = NMS(faces, true, 0.8);
		faces = PCN_3(paddedImg, img180, img90, imgNeg90, net_3, thresholds[2], 48, faces);
		faces = NMS(faces, false, 0.3);

		tm.stop();
		std::cout << "Time Cost: " << tm.getTimeMilli() << " ms" << std::endl;

		std::vector<FaceBox> preList = TransformBoxes(img, paddedImg, faces);

		for (int i = 0; i < preList.size(); i++)
		{
			DrawFace(img, preList[i]);
            int prediction = 10;
            // // And finally write all we've found out to the original image!
            // // First of all draw a green rectangle around the detected face:
            // rectangle(img, face_i, CV_RGB(0, 255,0), 1);
            // // Create the text we will annotate the box with:

		}

		cv::imshow("IMG", img);
		cv::waitKey(33);
	}
   
    return 1;

}


// vector<Mat> images;
// vector<std::string> labels;
// // Read in the data. This can fail if no valid

// Mat testSample = images[images.size() - 1];

// images.pop_back();
// labels.pop_back();

// Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
// model->train(images, labels);
// // The following line predicts the label of a given
// // test image:
// std::string predictedLabel = model->predict(testSample);

// double confidence = 0.0;
// model->predict(testSample, predictedLabel, confidence);
