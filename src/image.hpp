#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

namespace graphic
{
  class Image
  {
  public:
    Image() = default;
    Image(const Image& other) : m_mat(other.mat()) {};
    Image& operator=(const Image& other) 
    {
      if (this != &other)
      {
        m_mat = other.m_mat;
      }

      return *this;
    };
    Image (cv::Size size, int type, const cv::Scalar &s) : m_mat(size, type, s) {};
    ~Image() = default;

    void resize(float _scale);
    cv::Mat pad() const;

    cv::Mat preprocess();
    cv::Mat preprocess(int _dim);

    cv::Mat mat() const;
    void* data() const;

  private:
    cv::Mat m_mat;
  };
}