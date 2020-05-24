#pragma once

#include "image.hpp"
#include "constants.hpp"
#include <opencv2/face.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace math
{
  using Mat = cv::Mat;
  using Image = graphic::Image;

  struct FaceBox
  {
    int x, y, w, h;
    float angle, scale, score;

    cv::Rect getBox() { return cv::Rect(x,y,w,h); }

    FaceBox(int x_, int y_, int w_, int h_, float a_, float s_, float c_)
    : x(x_), y(y_), w(w_), h(h_), angle(a_), scale(s_), score(c_)
    {}
  };

  class Algorithm
  {
  public:
    Algorithm() = delete;
    Algorithm(const Algorithm&) = delete;
    Algorithm(Algorithm&&) = delete;
    Algorithm& operator=(const Algorithm&) = delete;
    Algorithm& operator=(Algorithm&&) = delete;
    ~Algorithm() = delete;

    static inline bool xyValid(int _x, int _y, Image _img)
    {
      return (_x >= 0 && _x < _img.mat().cols && _y >= 0 && _y < _img.mat().rows);
    }

    static float IoU(FaceBox &box1, FaceBox &box2);
    static std::vector<FaceBox> NMS(std::vector<FaceBox> &_faces, bool _local, float _threshold);
    static std::vector<FaceBox> TransformBoxes(Image _img, Image _imgPad, std::vector<FaceBox> &_faces);

    static std::vector<FaceBox> PCN_1(Image _img, Image _paddedImg, cv::dnn::Net _net, float _thresh, int _minFaceSize);
    static std::vector<FaceBox> PCN_2(Image _img, Image _img180, cv::dnn::Net _net, float _threshold, int _dim, std::vector<FaceBox> _faces);
    static std::vector<FaceBox> PCN_3(Image _img, Image _img180, Image _img90, Image _imgNeg90, cv::dnn::Net _net, float _threshold, int _dim, std::vector<FaceBox> _faces);
  };
}