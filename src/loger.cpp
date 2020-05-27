#include "loger.hpp"
#include <iostream>

namespace loger
{
  void Info(const std::string& s)
  {
    std::cout << "Info: " << s << std::endl;
  }

  void Error(const std::string& s)
  {
    std::cout << "Info: " << s << std::endl;
  }
}