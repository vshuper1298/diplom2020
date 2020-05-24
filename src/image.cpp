#include "image.hpp"

namespace graphic
{

  Image Image::preprocess()
  {
      cv::Mat mean(m_mat.size(), CV_32FC3, cv::Scalar(104, 117, 123));
      cv::Mat imgF;
      m_mat.convertTo(imgF, CV_32FC3);
      return Image(imgF - mean);
  }

  Image Image::preprocess(int _dim)
  {
      cv::Mat imgNew;
      cv::resize(m_mat, imgNew, cv::Size(_dim, _dim));
      cv::Mat mean(imgNew.size(), CV_32FC3, cv::Scalar(104, 117, 123));
      cv::Mat imgF;
      imgNew.convertTo(imgF, CV_32FC3);
      return Image(imgF - mean);

      // cv::resize(m_mat, m_mat, cv::Size(_dim, _dim));
      // m_mat.convertTo(imgF, CV_32FC3);
  }


  void Image::resize(float _scale)
  {
    cv::resize(m_mat, m_mat, cv::Size(int(m_mat.cols / _scale), int(m_mat.rows / _scale)));
  }

  Image Image::pad() const
  {
    int row = std::min(int(m_mat.rows * 0.2), 100);
    int col = std::min(int(m_mat.cols * 0.2), 100);
    cv::Mat aux;
    cv::copyMakeBorder(m_mat, aux, row, row, col, col, cv::BORDER_CONSTANT);
    return Image(aux);
  }

  const cv::Mat& Image::mat() const
  {
    return m_mat;
  }

  char* Image::data() const
  {
    return (char*)m_mat.data; //TODO: Change cast
  }
}