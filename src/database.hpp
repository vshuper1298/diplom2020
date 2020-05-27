#pragma once

#include "config.hpp"
#include "loger.hpp"
#include <sqlite3.h>

namespace storage
{
  class DataBase
  {
  public:
    DataBase() = default;
    ~DataBase() = default;

    DataBase(const DataBase& other) = default;
    DataBase& operator=(const DataBase& other) = default;

    DataBase(DataBase&& other) = default;
    DataBase& operator=(DataBase&& other) = default;

    bool checkResult(int rc, char* errMsg);

    bool openDB();
    void closeDB();

    bool executeQuery(const char *query);

    bool select();
    bool insert(const std::string& data);
    bool update();
    bool Delete();

  private:
    // std::map<int,int> m_faces = nullptr;
    sqlite3* m_database = nullptr;
  };
}