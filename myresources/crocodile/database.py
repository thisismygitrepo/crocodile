
import time
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
        self.con = None
        self.ses = None
        self.insp = None
        self.meta = None
        self.path = tb.P(self.eng.url.database)

        self.db = db
        self.sch = sch
        self.vws = vws
        self.schema = None
        # self.tables = None
        # self.views = None
        self.sch_tab = None
        self.sch_vws = None
        self.refresh()

    def refresh(self, sch=None):  # fails if multiple schemas are there and None is specified
        self.con = self.eng.connect()
        self.ses = sessionmaker()(bind=self.eng)  # ORM style
        self.meta = MetaData()
        self.meta.reflect(bind=self.eng, schema=sch or self.sch)
        self.insp = inspect(subject=self.eng)
        self.schema = tb.L(self.insp.get_schema_names())
        self.sch_tab = tb.Struct.from_keys_values(self.schema, self.schema.apply(lambda x: self.insp.get_table_names(schema=x)))
        self.sch_vws = tb.Struct.from_keys_values(self.schema, self.schema.apply(lambda x: self.insp.get_view_names(schema=x)))
        return self

    def __getstate__(self): return tb.Struct(self.__dict__.copy()).delete(keys=["eng", "con", "ses", "insp", "meta"]).update(path=self.path.collapseuser()).__dict__
    def __setstate__(self, state): self.__dict__.update(state); self.eng = self.make_sql_engine(self.path); self.refresh()

    @classmethod
    def from_local_db(cls, path=None, echo=False): return cls(engine=cls.make_sql_engine(path=path, echo=echo))
    def __repr__(self): return f"DataBase @ {self.eng}"
    def get_columns(self, table, sch=None): return self.meta.tables[self._get_table_identifier(table, sch)].exported_columns.keys()
    def close(self, sleep=2): print(f"Terminating database `{self.path.as_uri()}`"); self.con.close(); self.ses.close(); self.eng.dispose(); time.sleep(sleep)
    def _get_table_identifier(self, table, sch):
        if sch is None: sch = self.sch
        if sch is not None: return sch + "." + table
        else: return table

    @staticmethod
    def make_sql_engine(path=None, echo=False, dialect="sqlite", driver=["pysqlite", "DBAPI"][0]):
        """Establish lazy initialization with database"""
        if path == "memory":
            print("Linking to in-memory database.")
            from sqlalchemy.pool import StaticPool  # see: https://docs.sqlalchemy.org/en/14/dialects/sqlite.html#using-a-memory-database-in-multiple-threads
            return create_engine(url=f"{dialect}+{driver}:///:memory:", echo=echo, future=True, poolclass=StaticPool, connect_args={"check_same_thread": False})
        path = tb.P.tmpfile(folder="tmp_dbs", suffix=".db") if path is None else tb.P(path).expanduser().absolute().create(parents_only=True)
        print(f"Linking to database at {path.as_uri()}")
        return create_engine(url=f"{dialect}+{driver}:///{path}", echo=echo, future=True)  # echo flag is just a short for the more formal way of logging sql commands.

    # ==================== QUERIES =====================================
    def execute_as_you_go(self, *commands, res_func=lambda x: x.all(), df=False):
        with self.eng.connect() as conn:
            for command in commands: result = conn.execute(text(command))
            conn.commit()  # if driver is sqlite3, the connection is autocommitting. # this commit is only needed in case of DBAPI driver.
        return res_func(result) if not df else pd.DataFrame(res_func(result))

    def execute_begin_once(self, command, res_func=lambda x: x.all(), df=False):
        with self.eng.begin() as conn:
            result = conn.execute(text(command))  # no need for commit regardless of driver
            result = res_func(result)
        return result if not df else pd.DataFrame(result)

    def execute(self, command, df=False):
        with self.eng.begin() as conn: result = conn.execute(text(command))
        return result if not df else pd.DataFrame(result)

    def execute_script(self, command, df=False):
        with self.eng.begin() as conn: result = conn.executescript(text(command))
        return result if not df else pd.DataFrame(result)

    # ========================== TABLES =====================================
    def read_table(self, table, sch=None, size=100):
        res = self.con.execute(text(f'''SELECT * FROM "{self._get_table_identifier(table, sch)}"'''))
        return pd.DataFrame(res.fetchmany(size))

    def insert_dicts(self, table, *mydicts):
        cmd = f"""INSERT INTO {table} VALUES """
        for mydict in mydicts: cmd += f"""({tuple(mydict)}), """
        self.execute_begin_once(cmd)

    def describe_table(self, table, sch=None, dtype=True):
        print(table.center(100, "="))
        self.refresh()
        tbl = self.meta.tables[table]
        count = self.ses.query(tbl).count()
        res = tb.Struct(name=table, count=count, size_mb=count * len(tbl.exported_columns) * 10 / 1e6)
        res.print(dtype=False, as_config=True)
        dat = self.read_table(table=table, sch=sch, size=2)
        cols = self.get_columns(table, sch=sch)
        df = pd.DataFrame.from_records(dat, columns=cols)
        print("SAMPLE:\n", df)
        if dtype: print("\nDETAILED COLUMNS:\n", pd.DataFrame(self.insp.get_columns(table)))
        print("\n" * 3)


if __name__ == '__main__':
    pass
