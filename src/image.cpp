#include "image.hpp"

namespace graphic
{

  cv::Mat Image::preprocess()
  {
      cv::Mat mean(m_mat.size(), CV_32FC3, cv::Scalar(104, 117, 123));
      cv::Mat imgF;
      m_mat.convertTo(imgF, CV_32FC3);
      return imgF - mean;
  }

  cv::Mat Image::preprocess(int _dim)
  {
      cv::Mat imgNew;
      cv::resize(m_mat, imgNew, cv::Size(_dim, _dim));
      cv::Mat mean(imgNew.size(), CV_32FC3, cv::Scalar(104, 117, 123));
      cv::Mat imgF;
      imgNew.convertTo(imgF, CV_32FC3);
      return imgF - mean;

      // cv::resize(m_mat, m_mat, cv::Size(_dim, _dim));
      // m_mat.convertTo(imgF, CV_32FC3);
  }


  void Image::resize(float _scale)
  {
    cv::resize(m_mat, m_mat, cv::Size(int(m_mat.cols / _scale), int(m_mat.rows / _scale)));
  }

  cv::Mat Image::pad() const
  {
    int row = std::min(int(m_mat.rows * 0.2), 100);
    int col = std::min(int(m_mat.cols * 0.2), 100);
    cv::Mat aux;
    cv::copyMakeBorder(m_mat, aux, row, row, col, col, cv::BORDER_CONSTANT);
    return aux;
  }

  cv::Mat Image::mat() const
  {
    return m_mat;
  }

  void* Image::data() const
  {
    return m_mat.data;
  }
}