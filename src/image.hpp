#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

namespace graphic
{
  class Image
  {
  public:
    Image() = default;
    Image(const cv::Mat& m) : m_mat(m) {};
    Image (cv::Size size, int type, const cv::Scalar &s) : m_mat(size, type, s) {};
    Image(const Image& other) : m_mat(other.mat()) {};
    Image& operator=(const Image& other) 
    {
      if (this != &other)
      {
        m_mat = other.m_mat;
      }

      return *this;
    };

    ~Image() = default;

    Image& operator-(const Image& other)
    {
      m_mat -= other.m_mat;
      return *this;
    }

    cv::Mat operator()(const cv::Rect& roi) const
    {
      return m_mat(roi);
    }

    void resize(float _scale);
    Image pad() const;

    Image preprocess();
    Image preprocess(int _dim);

    int rows() const { return m_mat.rows; };
    int cols() const { return m_mat.cols; };

    const cv::Mat& mat() const;
    cv::Mat mat() {return m_mat;};
    char* data() const;

  private:
    cv::Mat m_mat;
  };
}