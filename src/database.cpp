#include "database.hpp"

namespace storage
{
  static int callback(void *data, int argc, char **argv, char **azColName)
  {
    loger::Info(static_cast<const char*>(data));

    for (int i = 0; i < argc; i++)
    {
      loger::Info(std::string(azColName[i]) + " = " + std::string(argv[i] ? argv[i] : "NULL"));
    }
    
    return 0;
  }

  bool DataBase::checkResult(int rc, char* errMsg)
  {
    if( rc != SQLITE_OK ) 
    {
      loger::Error(std::string("SQL error: ") + std::string(errMsg));
      sqlite3_free(errMsg);
      return false;
    } 
    else 
    {
      loger::Info("Operation done successfully");
      return true;
    }
  }

  bool DataBase::openDB()
  {
    /* Open database */
    int rc = sqlite3_open(config::storage::DATABASE_FILE, &m_database);

    if(rc)
    {
      loger::Error(std::string("Can't open database: ") + std::string(sqlite3_errmsg(m_database)));
      return false;
    }
    else
    {
      loger::Info("DataBase oppened done successfully");
      return true;
    }    
  }

  void DataBase::closeDB()
  {
    sqlite3_close(m_database);
  }

  bool DataBase::executeQuery(const char *query)
  {
    char *zErrMsg = 0;
    const char* data = "Callback function called";

    openDB();
    int rc = sqlite3_exec(m_database, query, callback, (void*)data, &zErrMsg);
    bool bres = checkResult(rc, zErrMsg);
    closeDB();
    return bres;
  }

  bool DataBase::select()
  {
    /* Create and Execute SQL statement */
    const char *query = "SELECT * from Images";
    return executeQuery(query);
  }

  bool DataBase::insert(const std::string& data) //REFACTOR!!!!!!!!!!!!!!!
  {
    /* Create and Execute SQL statement */
    const char *query = "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) "  \
        "VALUES (1, 'Paul', 32, 'California', 20000.00 ); " \
         "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) "  \
         "VALUES (2, 'Allen', 25, 'Texas', 15000.00 ); "     \
         "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)" \
         "VALUES (3, 'Teddy', 23, 'Norway', 20000.00 );" \
         "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)" \
         "VALUES (4, 'Mark', 25, 'Rich-Mond ', 65000.00 );";
    return executeQuery(query);
  }

  bool DataBase::update() //REFACTOR!!!!!!!!!!!!!!!
  {
    /* Create and Execute SQL statement */
    const char *query = "UPDATE COMPANY set SALARY = 25000.00 where ID=1; ";
    return executeQuery(query);
  }

  bool DataBase::Delete()
  {
    /* Create and Execute SQL statement */
    const char *query = "DELETE from COMPANY where ID=2; ";
    return executeQuery(query);
  }
}