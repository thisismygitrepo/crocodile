

from sqlalchemy.sql import schema
import conn.servers as servers
import sqlalchemy
from sqlalchemy.engine import URL
from sqlalchemy.sql.schema import MetaData
from sqlalchemy import inspect
import crocodile.toolbox as tb
import pandas as pd


class Server:
    def __init__(self) -> None:
        pass


class DB:
    def __init__(self, sv, db=None, sch=None, views=True):
        self.engine = get_engine(sv, db)
        self.sch = sch
        self.conn = self.engine.connect()
        self.sv = sv
        self.meta = MetaData(self.conn, schema=self.sch)  # can fail if there are multiple schemas
        self.meta.reflect(views=views)
        self.insp = inspect(self.engine)
        # self.c = self.conn.cursor()
        self.schema = tb.L(self.insp.get_schema_names())
        self.tables = self.schema.apply(lambda x: self.insp.get_table_names(schema=x))
        self.views = self.schema.apply(lambda x: self.insp.get_view_names(schema=x))
        self.sch_tab = tb.Struct.from_keys_values(self.schema, self.tables)
        self.sch_vws = tb.Struct.from_keys_values(self.schema, self.views)

    def get_table_str(self, table, sch):
        if sch is None: sch = self.sch
        if sch is not None: 
            return sch + "." + table
        else:
            return table

    def read_table(self, table, sch=None, size=100):
        # with self.conn:
        cols = f"TOP ({size})" if size is not None else ""
        res = self.conn.execute(f"""SELECT {cols} * FROM {self.get_table_str(table, sch)}""")
        return res.fetchall()

    def get_columns(self, table, sch=None):
        return self.meta.tables[self.get_table_str(table, sch)].exported_columns.keys()

    def read_df(self, table, schema=None):
        df = pd.read_sql_table(table, con=self.engine, schema=schema or self.sch)
        return df
        
    def describe_table(self, table, sch=None, dtype=True):
        print(table.center(100, "="))
        tmp = self.conn.execute(f"EXEC sp_spaceused N'{self.get_table_str(table, sch)}'").fetchall()[0]
        res = tb.Struct(name=tmp[0], num_rows=int(tmp[1]) if type(tmp[1]) is str else "None", size_mb=tmp[2])
        res.print(dtype=False)
        self.meta.reflect(self.conn, schema="exp")
        dat = self.read_table(table=table, sch=sch, size=2)
        cols = self.get_columns(table, sch=sch)
        df = pd.DataFrame.from_records(dat, columns=cols)
        print("SAMPLE:\n", df)
        if dtype:
            print("\n")
            print("DETAILED COLUMNS:\n", list(self.meta.tables[self.get_table_str(table, sch)].columns))
        print("\n" * 3)

    @staticmethod
    def list_databases(conn):
        """Connection must not point to a database, but rather sever as a whole"""
        rows = conn.execute("select name FROM sys.databases;")
        return [row["name"] for row in rows]


def get_engine(sv, db=None):
    """
    Alchemy format: takes in URL string.
    engine = sqlalchemy.create_engine('mssql+pyodbc://user:password@DSN')
    A DSN defines server/database.  
    Internally, this URL will be traslated to connection string and sent off to pyodbc
    That said, there is a large number of errors that can happen when the URL is translated to connection string.
    For examply, if password has `@` in it, then the parser will think this is the point for giving server name. etc
    For this reason it is optimal to pass

    Direct format:
    # engine = sqlalchemy.create_engine(f'mssql+pyodbc://{sv.username}:{sv.password}@{sv.servername}?driver=ODBC+Driver+17+for+SQL+Server')

    Reference:
    https://docs.sqlalchemy.org/en/14/dialects/mssql.html#module-sqlalchemy.dialects.mssql.pyodbc
    https://stackoverflow.com/questions/15750711/connecting-to-sql-server-2012-using-sqlalchemy-and-pyodbc
    """

    db_string = f"DATABASE={db};" if db is not None else ""
    connection_string = r'DRIVER={ODBC Driver 17 for SQL Server};' + fr'SERVER={sv.servername};' + db_string + sv.odbc_creds
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
    engine = sqlalchemy.create_engine(connection_url)
    return engine



def test_udbc():
    sv =  servers.Clinipi()
    db = "ClinEpiReporting"

    """
    `pyodbc` is the Python version of ODBC driver required to connect to the machine Microsoft SQL server.
    * "TRUSTED_CONNECTION=YES"  means use windows login details.
    * no spaces around `=`, ecpecially on Windows machines.

    """
    import pyodbc
    conn_str = r'DRIVER={ODBC Driver 17 for SQL Server};' + fr'SERVER={sv.servername};DATABASE={db};UID={sv.username};PWD={sv.password}'
    print(conn_str)
    conn = pyodbc.connect(conn_str)
    return conn


def main():
    sv =  servers.Clinipi()
    db = "ClinEpiReporting"
    return get_engine(sv, db)


if __name__ == "__main__":
    pass
