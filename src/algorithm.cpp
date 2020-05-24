#include "algorithm.hpp"

namespace math
{
  float Algorithm::IoU(FaceBox &box1, FaceBox &box2)
  {
    int xOverlap = std::max(0, std::min(box1.x + box1.w - 1, box2.x + box2.w - 1) - std::max(box1.x, box2.x) + 1);
    int yOverlap = std::max(0, std::min(box1.y + box1.h - 1, box2.y + box2.h - 1) - std::max(box1.y, box2.y) + 1);
    int intersection = xOverlap * yOverlap;
    int unio = box1.w * box1.h + box2.w * box2.h - intersection;

    return float(intersection) / unio;
  }

  std::vector<FaceBox> Algorithm::NMS(std::vector<FaceBox> &_faces, bool _local, float _threshold)
  {
    if (_faces.size() == 0)
        return _faces;

    std::sort(_faces.begin(), _faces.end(), [](const FaceBox &b1, const FaceBox &b2){ return b1.score > b2.score;});
    std::vector<int> flag(_faces.size(), 0);
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

  std::vector<FaceBox> Algorithm::TransformBoxes(Image _img, Image _imgPad, std::vector<FaceBox> &_faces)
  {
    int row = (_imgPad.rows() - _img.rows()) / 2;
    int col = (_imgPad.cols() - _img.cols()) / 2;

    std::vector<FaceBox> newFaceBoxes;
    for(int i = 0; i < _faces.size(); i++)
    {
      if (_faces[i].w > 0 && _faces[i].h > 0)
        newFaceBoxes.push_back(FaceBox(_faces[i].x - col, _faces[i].y - row, _faces[i].w, _faces[i].w, _faces[i].angle, _faces[i].scale, _faces[i].score));
    }
    return newFaceBoxes;
  }

  std::vector<FaceBox> Algorithm::PCN_1(Image _img, Image _paddedImg, cv::dnn::Net _net, float _thresh, int _minFaceSize)
  {
    std::vector<FaceBox> faceBoxes_1;
    int row = (_paddedImg.rows() - _img.rows()) / 2;
    int col = (_paddedImg.cols() - _img.cols()) / 2;
    int netSize = 24;
    int minFace = _minFaceSize * 1.4; // - size 20 + 40%
    float currentScale = minFace / float(netSize);
    int stride = 8;

    graphic::Image resizedImg = _img;
    resizedImg.resize(currentScale);

    while (std::min(resizedImg.rows(), resizedImg.cols()) >= netSize)
    {
      // - Set input for net
      graphic::Image inputImage = resizedImg.preprocess();
      cv::Mat inputBlob = cv::dnn::blobFromImage(inputImage.mat(), 1.0, cv::Size(), cv::Scalar(), false, false);
      _net.setInput(inputBlob);

      std::vector<cv::String> outputBlobNames = {"cls_prob", "rotate_cls_prob", "bbox_reg_1" };
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
          {
            continue;
          }

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

      resizedImg.resize(1.414);
      currentScale = float(_img.rows()) / resizedImg.rows();
    }

    return faceBoxes_1;
  }

  std::vector<FaceBox> Algorithm::PCN_2(Image _img, Image _img180, cv::dnn::Net _net, float _threshold, int _dim, std::vector<FaceBox> _faces)
  {
    //_dim = 24 ---> network size
    if (_faces.size() == 0)
      return _faces;

    std::vector<Image> dataList;
    int height = _img.rows();
    for (int i = 0; i < _faces.size(); i++)
    {
      if (abs(_faces[i].angle) < EPS)
      {
        Image tmp = _img(_faces[i].getBox());
        dataList.push_back(tmp.preprocess(_dim));
      }
      else
      {
        int y2 = _faces[i].y + _faces[i].h - 1;
        Image tmp = _img180(cv::Rect(_faces[i].x, height - 1 - y2, _faces[i].w, _faces[i].h));
        dataList.push_back(tmp.preprocess(_dim));
      }
    }

    std::vector<FaceBox> faceBoxes_2;

    // - Process the dataList vector
    for(int dataNr=0; dataNr<dataList.size(); dataNr++)
    {
      Mat inputBlob = cv::dnn::blobFromImage(dataList[dataNr].mat(), 1.0, cv::Size(), cv::Scalar(), false, false);
      _net.setInput(inputBlob);
      std::vector<cv::String> outputBlobNames = {"cls_prob", "rotate_cls_prob", "bbox_reg_2" };
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

  std::vector<FaceBox> Algorithm::PCN_3(Image _img, Image _img180, Image _img90, Image _imgNeg90, cv::dnn::Net _net, float _threshold, int _dim, std::vector<FaceBox> _faces)
  {
    if (_faces.size() == 0)
      return _faces;

    std::vector<Image> dataList;
    int height = _img.rows();
    int width = _img.cols();
    for (int i = 0; i < _faces.size(); i++)
    {
      if (abs(_faces[i].angle) < EPS)
      {
        Image tmp = _img(cv::Rect(_faces[i].x, _faces[i].y, _faces[i].w, _faces[i].h));
        dataList.push_back(tmp.preprocess(_dim));
      }
      else if (abs(_faces[i].angle - 90) < EPS)
      {
        Image tmp = _img90(cv::Rect(_faces[i].y, _faces[i].x, _faces[i].h, _faces[i].w));
        dataList.push_back(tmp.preprocess(_dim));
      }
      else if (abs(_faces[i].angle + 90) < EPS)
      {
        int x = _faces[i].y;
        int y = width - 1 - (_faces[i].x + _faces[i].w - 1);
        Image tmp = _imgNeg90(cv::Rect(x, y, _faces[i].w, _faces[i].h));
        dataList.push_back(tmp.preprocess(_dim));
      }
      else
      {
        int y2 = _faces[i].y + _faces[i].h - 1;
        Image tmp = _img180(cv::Rect(_faces[i].x, height - 1 - y2, _faces[i].w, _faces[i].h));
        dataList.push_back(tmp.preprocess(_dim));
      }
    }

    std::vector<FaceBox> faceBoxes_3;

    for(int dataNr=0; dataNr<dataList.size(); dataNr++)
    {
      Mat inputBlob = cv::dnn::blobFromImage(dataList[dataNr].mat(), 1.0, cv::Size(), cv::Scalar(), false, false);
      _net.setInput(inputBlob);
      std::vector<cv::String> outputBlobNames = {"cls_prob", "rotate_reg_3", "bbox_reg_3" };
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
      Image imgTmp = _img;
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
        {
          faceBoxes_3.push_back(FaceBox(x, y, w, w, angle, _faces[dataNr].scale, score));
        }
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
}