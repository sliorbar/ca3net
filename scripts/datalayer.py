from pandas.io.formats.style_render import Subset
import pyodbc
import urllib
import os
import datetime
from pandas import DataFrame as df
from sqlalchemy import create_engine
from sympy import false

def _load_dotenv():
    """
    Load key=value pairs from a local .env file without requiring extra packages.
    Existing environment variables are preserved.
    """
    candidate_paths = [
        os.path.join(os.getcwd(), "ca3net.env"),
        os.path.join(os.path.dirname(__file__), "ca3net.env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "ca3net.env"),
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    ]

    for env_path in candidate_paths:
        if not os.path.exists(env_path):
            continue

        with open(env_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
        break


def InitializeSQLEngine():
    """
    Create a new instance of sql engine to log experiment data
    """
    _load_dotenv()

    WINDOWS_HOST = os.environ.get("OMEN_DB_HOST")
    DB = os.environ.get("OMEN_DB_NAME", "CUNY")
    USER = os.environ.get("OMEN_DB_USER")
    PWD = os.environ.get("OMEN_DB_PASSWORD")

    if not USER or not PWD:
        raise ValueError(
            "Missing database credentials. Set OMEN_DB_USER and "
            "OMEN_DB_PASSWORD in your environment or .env file."
        )

# Create the connection string
    odbc_str = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={WINDOWS_HOST},1433;"
    f"DATABASE={DB};"
    f"UID={USER};PWD={PWD};"
    "Encrypt=yes;TrustServerCertificate=yes;"
    )

    params = urllib.parse.quote_plus(odbc_str)

    engine = create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        fast_executemany=True,
)
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