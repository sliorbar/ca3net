from pandas.io.formats.style_render import Subset
import pyodbc
import urllib
import datetime
from pandas import DataFrame as df
from sqlalchemy import create_engine
from sympy import false
def InitializeSQLEngine():
    """
    Create a new instance of sql engine to log experiment data
    """

    engine = create_engine("mssql+pyodbc://lior_cuny:!CUNEWyork2019@CUNY2",fast_executemany=True)
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
        conn.close()
    return 

def SaveTrial(engine,data, tablename,expid, selected_pc = None,unpivot=False, offset=0, dfIndex=None):
    """
    Create a new entry to log experiment data
    """
    #engine1 = InitializeSQLEngine()
    
    
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
    savedata['SelectedPC']=selected_pc
    savedata['expid'] = expid
    conn = engine.connect()
    message = 'Writing to database %d rows - %s' % (savedata.shape[0], tablename)
    print(message)
    savedata.to_sql(name=tablename,con=conn,if_exists='append',chunksize=200)
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
        
        conn.close()
    return query_result