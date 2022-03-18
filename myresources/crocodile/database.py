

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.sql.schema import MetaData
import crocodile.toolbox as tb
import pandas as pd


"""
https://docs.sqlalchemy.org/en/14/tutorial/index.html#a-note-on-the-future

"""

tb.Display.set_pandas_display()


class DBMS:
    """Implementation Philosophy:
    * Always use sqlalchemy API and avoid sql-dielect specific language.
    * Engine is provided externally. It is the end-user's business to make this engine.
    """
    def __init__(self, engine, db=None, sch=None, vws=False):
        self.eng = engine
        self.path = tb.P(self.eng.url.database)
        self.con = self.eng.connect()
        self.ses = sessionmaker()(bind=self.eng)  # ORM style
        self.db = db
        self.sch = sch
        self.vws = vws

        self.insp = None
        self.meta = MetaData()
        self.schema = None
        self.tables = None
        self.views = None
        self.sch_tab = None
        self.sch_vws = None
        self.refresh()

    def close(self):
        self.con.close()
        self.ses.close()
        self.eng.dispose()

    def refresh(self, sch=None):
        # fails if multiple schemas are there and None is specified
        self.meta.reflect(bind=self.eng, schema=sch or self.sch)
        self.insp = inspect(subject=self.eng)

        self.schema = tb.L(self.insp.get_schema_names())
        self.schema.append(None)
        self.tables = self.schema.apply(lambda x: self.insp.get_table_names(schema=x))
        # self.tables = [self.meta.tables[tmp] for tmp in self.meta.tables.keys()]
        self.views = self.schema.apply(lambda x: self.insp.get_view_names(schema=x))
        self.sch_tab = tb.Struct.from_keys_values(self.schema, self.tables)
        self.sch_vws = tb.Struct.from_keys_values(self.schema, self.views)

        return self

    @classmethod
    def from_local_db(cls, path=None, echo=False):
        return cls(engine=cls.make_sql_db(path, echo))

    def __repr__(self):
        return f"DataBase @ {self.eng}"

    @staticmethod
    def make_sql_db(path=None, echo=False, dialect="sqlite", driver=["pysqlite", "DBAPI"][0]):
        """Establish lazy initialization with database"""
        # core style, use in conjustction with Connect.
        if path == "memory":
            return create_engine(url=f"{dialect}+{driver}:///:memory:", echo=echo, future=True)
        if path is None:
            path = tb.P.tmpfile(folder="tmp_dbs", suffix=".db")
        print(f"Linking to database at {tb.P(path).as_uri()}")
        eng = create_engine(url=f"{dialect}+{driver}:///{path}", echo=echo, future=True)
        # echo flag is just a short for the more formal way of logging sql commands.
        return eng

    # ==================== QUERIES =====================================
    def execute_as_you_go(self, *commands, res_func=lambda x: x.all()):
        with self.eng.connect() as conn:
            for command in commands:
                result = conn.execute(text(command))
            conn.commit()  # if driver is sqlite3, the connection is autocommitting.
            # this commit is only needed in case of DBAPI driver.
        return res_func(result)

    def execute_begin_once(self, command, res_func=lambda x: x.all()):
        with self.eng.begin() as conn:
            result = conn.execute(text(command))
            # no need for commit regardless of driver
            result = res_func(result)
        return result

    def execute(self, command):
        with self.eng.begin() as conn:
            result = conn.execute(text(command))
        return result

    def _get_table_identifier(self, table, sch):
        if sch is None: sch = self.sch
        if sch is not None:
            return sch + "." + table
        else:
            return table

    # ========================== TABLES =====================================
    def read_table(self, table, sch=None, size=100):
        res = self.con.execute(text(f"""SELECT * FROM {self._get_table_identifier(table, sch)}"""))
        return res.fetchmany(size)

    def make_df(self, table_name, records=None, schema=None):
        self.meta.reflect(bind=self.eng, schema=schema or self.sch)
        table = self.meta.tables[table_name]
        res = pd.DataFrame(records or self.ses.query(table).all(), columns=table.exported_columns.keys())
        # the following spits an error if sqlalchemy is 2.0
        # df = pd.read_sql_table(table, con=self.eng, schema=schema or self.sch)
        return res

    def get_columns(self, table, sch=None):
        return self.meta.tables[self._get_table_identifier(table, sch)].exported_columns.keys()

    def insert_dicts(self, table, *mydicts):
        cmd = f"""INSERT INTO {table} VALUES """
        for mydict in mydicts:
            cmd += f"""({tuple(mydict)}), """
        self.execute_begin_once(cmd)

    def describe_table(self, table, sch=None, dtype=True):
        print(table.center(100, "="))
        self.refresh()
        tbl = self.meta.tables[table]
        count = self.ses.query(tbl).count()
        res = tb.Struct(name=table , count=count,
                        size_mb=count * len(tbl.exported_columns) * 10 / 1e6)
        res.print(dtype=False, config=True)
        dat = self.read_table(table=table, sch=sch, size=2)
        cols = self.get_columns(table, sch=sch)
        df = pd.DataFrame.from_records(dat, columns=cols)
        print("SAMPLE:\n", df)
        if dtype:
            print("\n")
            print("DETAILED COLUMNS:\n", tb.pd.DataFrame(self.insp.get_columns(table)))
            # print("DETAILED COLUMNS:\n", list(self.meta.tables[self._get_table_identifier(table, sch)].columns))
        print("\n" * 3)


if __name__ == '__main__':
    pass
