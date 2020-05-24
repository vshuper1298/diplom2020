#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv4/opencv2/contrib.hpp>
#include <iostream>
#include <thread>
#include "algorithm.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace math;

#define M_PI  3.14159265358979323846
#define EPS  1e-5


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

void DrawFace(cv::Mat img, math::FaceBox face)
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

vector<graphic::Image> images;
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

	
cv::Mat matt;
	while (!0)
	{
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		//read next frame
		if (!capture.read(matt))
		{
			std::cout << "Failed to read video!" << std::endl;
			return -1;
		}
    graphic::Image img(matt);
        images.push_back(img);
        // std::this_thread::sleep_for(std::chrono::seconds(10));
        if(images.size() == 2)
        {
            Mat testSample = images[images.size() - 1].mat();

            // images.pop_back();
            // labels.pop_back();

            Ptr<cv::face::FaceRecognizer> model = cv::face::EigenFaceRecognizer::create();//cv::face::createEigenFaceRecognizer();
            vector<cv::Mat> imgs;
            for (auto& v : images)
              imgs.push_back(v.mat());
            model->train(imgs, labels);
            // The following line predicts the label of a given
            // test image:
            cvtColor(testSample, testSample, COLOR_BGR2GRAY);

            int predictedLabel = model->predict(testSample);
            std::cout << "LABEL: " << predictedLabel << std::endl;

            double confidence = 0.0;
            model->predict(testSample, predictedLabel, confidence);
            std::cout << "CONF: " << confidence << std::endl;
        }

		cv::flip(img.mat(), img.mat(), 1);//Y-axis mirror (i.e. horizontal mirror)
		
		cv::Mat paddedImg = graphic::Image(img).pad().mat();
		cv::Mat img180, img90, imgNeg90;
		cv::flip(paddedImg, img180, 0);
		cv::transpose(paddedImg, img90);
		cv::flip(img90, imgNeg90, 0);

		float thresholds[] = { 0.37, 0.43, 0.95 };

		cv::TickMeter tm;
		tm.reset();
		tm.start();

		int minFaceSize = 40;
		std::vector<math::FaceBox> faces = math::Algorithm::PCN_1(img, paddedImg, net_1, thresholds[0], minFaceSize);
		faces = math::Algorithm::NMS(faces, true, 0.8);
		faces = math::Algorithm::PCN_2(paddedImg, img180, net_2, thresholds[1], 24, faces);
		faces = math::Algorithm::NMS(faces, true, 0.8);
		faces = math::Algorithm::PCN_3(paddedImg, img180, img90, imgNeg90, net_3, thresholds[2], 48, faces);
		faces = math::Algorithm::NMS(faces, false, 0.3);

		tm.stop();
		std::cout << "Time Cost: " << tm.getTimeMilli() << " ms" << std::endl;

		std::vector<math::FaceBox> preList = math::Algorithm::TransformBoxes(img, paddedImg, faces);

		for (int i = 0; i < preList.size(); i++)
		{
			DrawFace(img.mat(), preList[i]);
            int prediction = 10;
            // // And finally write all we've found out to the original image!
            // // First of all draw a green rectangle around the detected face:
            // rectangle(img, face_i, CV_RGB(0, 255,0), 1);
            // // Create the text we will annotate the box with:

		}

		cv::imshow("IMG", img.mat());
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
