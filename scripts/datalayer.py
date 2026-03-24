import datetime
import os
from pathlib import Path

import pyodbc
import urllib
from pandas import DataFrame as df
from pandas.io.formats.style_render import Subset
from sqlalchemy import create_engine
from sympy import false

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _parse_env_file(env_path):
    """Minimal .env parser so local config works even without python-dotenv."""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")

        if key and key not in os.environ:
            os.environ[key] = value


def _load_project_env():
    """Load environment variables from the project .env file when available."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        if load_dotenv is not None:
            load_dotenv(env_path)
        else:
            _parse_env_file(env_path)
    else:
        print(
            f"Warning: .env file is missing at {env_path}. "
            "Environment variables must be set in the shell."
        )


def _require_env(var_name):
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{var_name}'. "
            "Create a .env file in the project root or set it in your shell."
        )
    return value


def InitializeSQLEngine():
    """
    Create a new instance of sql engine to log experiment data
    """
    _load_project_env()

    server = _require_env("CA3NET_DB_SERVER")
    database = _require_env("CA3NET_DB_NAME")
    username = _require_env("CA3NET_DB_USER")
    password = _require_env("CA3NET_DB_PASSWORD")
    driver = os.getenv("CA3NET_DB_DRIVER", "ODBC Driver 17 for SQL Server")

    conn_str = (
        f"mssql+pyodbc://{username}:{password}@{server}/{database}"
        f"?driver={driver.replace(' ', '+')}"
    )

    engine = create_engine(conn_str, fast_executemany=True)
    return engine

def InitializeTrial(engine,description,details='test'):
    """
    Create a new entry to log experiment data
    """
    #engine1 = InitializeSQLEngine()
    from sqlalchemy import text
    dateTimeObj = datetime.datetime.now()
    dateTimeObj=dateTimeObj.replace(microsecond=0)
    querytext = "insert into Experiments (Description, [Param JSON],[Start Timestamp]) values( '%s','%s', '%s')" % (description,details,dateTimeObj)
    expridtext = "Select ID from Experiments where [Start Timestamp] ='%s'" % dateTimeObj
    with engine.connect() as conn:
        conn.execute(text(querytext))
        query_result = conn.execute(text(expridtext))
        for row in query_result:
            result = row[0]

        conn.commit()
        conn.close()
    return result

def UpdateTrial(engine,expid, description,details='test'):
    """
    Create a new entry to log experiment data
    """
    #engine1 = InitializeSQLEngine()
    from sqlalchemy import text
    #dateTimeObj = datetime.datetime.now()
    #dateTimeObj=dateTimeObj.replace(microsecond=0)
    querytext = "update Experiments set Description='%s',  [Param JSON] = '%s' where id=%s" % (description,details,expid)
    #expridtext = "Select ID from Experiments where [Start Timestamp] ='%s'" % dateTimeObj
    with engine.connect() as conn:
        conn.execute(text(querytext))
        #query_result = conn.execute(text(expridtext))
        conn.commit()
        conn.close()
    return 

def SaveTrial(engine,data, tablename,expid, selected_pc = None,unpivot=False, offset=0, dfIndex=None):
    """
    Create a new entry to log experiment data
    """
    #engine1 = InitializeSQLEngine()
    print('Saving data: ' + tablename)
    
    if unpivot:
        savedata = df(data,index=dfIndex)
        savedata=savedata.melt(ignore_index=False)
        #savedata=savedata.melt(ignore_index=False,id_vars='time_ms')
        savedata=savedata.rename(columns={'variable':'InputFromPC'})
        savedata.sort_index(inplace=True)
        savedata = savedata.drop_duplicates(subset = {'InputFromPC','value'})
        savedata['SelectedPC']=selected_pc
        savedata['offset'] = offset
        #savedata['time_ms'] = savedata['time_ms'].astype('string')
        #savedata=savedata.astype({'time_ms': 'float'})
        #savedata.fillna(0,inplace=True)
    
    
    else:
        savedata = df(data)
    if savedata.columns.size > 100:
        savedata=savedata.iloc[:,0:100]
    #savedata['SelectedPC']=selected_pc
    savedata['expid'] = expid
    conn = engine.connect()
    message = 'Writing to database %d rows - %s' % (savedata.shape[0], tablename)
    print(message)
    savedata.to_sql(name=tablename,con=conn,if_exists='append',chunksize=2000, index=False)
    conn.commit()
    print ('finished writing to database')
    conn.close()

    return 1

def CloseTrial(engine,expid = 0):
    """
    Create a new entry to log experiment data
    """
    #engine1 = InitializeSQLEngine()
    from sqlalchemy import text
    dateTimeObj = datetime.datetime.now()
    dateTimeObj=dateTimeObj.replace(microsecond=0)
    expridtext = "Update Experiments set [Finish Timestamp] ='%s' where ID ='%s'" % (dateTimeObj,str(expid))
    with engine.connect() as conn:
        query_result = conn.execute(text(expridtext))
        
        conn.commit()
        conn.close()
    return query_result
