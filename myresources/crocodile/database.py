
"""
https://docs.sqlalchemy.org/en/14/tutorial/index.html#a-note-on-the-future

"""

import time
from typing import Optional, Any, Callable

import pandas as pd

from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, text, inspect, Engine, Connection
from sqlalchemy.engine import Inspector
from sqlalchemy.sql.schema import MetaData
from crocodile.core import Struct, Display
from crocodile.file_management import List as L, P, OPLike

_ = create_engine, text


Display.set_pandas_display()


class DBMS:
    """Implementation Philosophy:
    * Always use sqlalchemy API and avoid sql-dielect specific language.
    * Engine is provided externally. It is the end-user's business to make this engine.
    """
    def __init__(self, engine: Engine, sch: Optional[str] = None, vws: bool = False):
        self.eng: Engine = engine
        self.con: Optional[Connection] = None
        self.ses: Optional[Session] = None
        self.insp: Optional[Inspector] = None
        self.meta: Optional[MetaData] = None
        self.path = P(self.eng.url.database) if self.eng.url.database else None  # memory db

        # self.db = db
        self.sch = sch
        self.vws: bool = vws
        self.schema: Optional[L[str]] = None
        # self.tables = None
        # self.views = None
        # self.sch_tab: Optional[Struct] = None
        # self.sch_vws: Optional[Struct] = None
        self.refresh()
        # self.ip_formatter: Optional[Any] = None
        # self.db_specs: Optional[Any] = None

    def refresh(self, sch: Optional[str] = None) -> 'DBMS':  # fails if multiple schemas are there and None is specified
        self.con = self.eng.connect()
        self.ses = sessionmaker()(bind=self.eng)  # ORM style
        self.meta = MetaData()
        self.meta.reflect(bind=self.eng, schema=sch or self.sch)
        insp = inspect(subject=self.eng)
        self.insp = insp
        self.schema = L(obj_list=self.insp.get_schema_names())
        self.sch_tab: dict[str, list[str]] = {k: v for k, v in zip(self.schema.list, self.schema.apply(lambda x: insp.get_table_names(schema=x)))}  # dict(zip(self.schema, self.schema.apply(lambda x: self.insp.get_table_names(schema=x))))  #
        self.sch_vws: dict[str, list[str]] = {k: v for k, v in zip(self.schema.list, self.schema.apply(lambda x: insp.get_view_names(schema=x)))}
        return self

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["con"]
        del state["ses"]
        del state["meta"]
        del state["insp"]
        del state["eng"]
        if self.path:
            state['path'] = self.path.collapseuser()
        return state
    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.eng = self.make_sql_engine(self.path)
        self.refresh()

    @classmethod
    def from_local_db(cls, path: OPLike = None, echo: bool = False, share_across_threads: bool = False, **kwargs: Any): return cls(engine=cls.make_sql_engine(path=path, echo=echo, share_across_threads=share_across_threads, **kwargs))
    def __repr__(self): return f"DataBase @ {self.eng}"
    def get_columns(self, table: str, sch: Optional[str] = None):
        assert self.meta is not None
        return self.meta.tables[self._get_table_identifier(table=table, sch=sch)].exported_columns.keys()
    def close(self, sleep: int = 2):
        if self.path:
            print(f"Terminating database `{self.path.as_uri() if 'memory' not in self.path else self.path}`")
        if self.con: self.con.close()
        if self.ses: self.ses.close()
        self.eng.pool.dispose()
        self.eng.dispose()
        time.sleep(sleep)
    def _get_table_identifier(self, table: str, sch: Optional[str]):
        if sch is None: sch = self.sch
        if sch is not None:
            return f"{sch}.'{table}'"
        else: return table

    @staticmethod
    def make_sql_engine(path: OPLike = None, echo: bool = False, dialect: str = "sqlite", driver: str = ["pysqlite", "DBAPI"][0], pool_size: int = 5, share_across_threads: bool = True, **kwargs: Any):
        """Establish lazy initialization with database"""
        if str(path) == "memory":
            print("Linking to in-memory database.")
            if share_across_threads:
                from sqlalchemy.pool import StaticPool  # see: https://docs.sqlalchemy.org/en/14/dialects/sqlite.html#using-a-memory-database-in-multiple-threads
                return create_engine(url=f"{dialect}+{driver}:///:memory:", echo=echo, future=True, poolclass=StaticPool, connect_args={"check_same_thread": False})
            else: return create_engine(url=f"{dialect}+{driver}:///:memory:", echo=echo, future=True, pool_size=pool_size, **kwargs)
        path = P.tmpfile(folder="tmp_dbs", suffix=".sqlite") if path is None else P(path).expanduser().absolute().create(parents_only=True)
        print(f"Linking to database at {path.as_uri()}")
        return create_engine(url=f"{dialect}+{driver}:///{path}", echo=echo, future=True, pool_size=10, **kwargs)  # echo flag is just a short for the more formal way of logging sql commands.

    # ==================== QUERIES =====================================
    def execute_as_you_go(self, *commands: str, res_func: Callable[[Any], Any] = lambda x: x.all(), df: bool = False):
        with self.eng.connect() as conn:
            result = None
            for command in commands:
                result = conn.execute(text(command))
            conn.commit()  # if driver is sqlite3, the connection is autocommitting. # this commit is only needed in case of DBAPI driver.
            return res_func(result) if not df else pd.DataFrame(res_func(result))

    def execute_begin_once(self, command: str, res_func: Callable[[Any], Any] = lambda x: x.all(), df: bool = False):
        with self.eng.begin() as conn:
            result = conn.execute(text(command))  # no need for commit regardless of driver
            result = res_func(result)
        return result if not df else pd.DataFrame(result)

    def execute(self, command: str, df: bool = False):
        with self.eng.begin() as conn: result = conn.execute(text(command))
        return result if not df else pd.DataFrame(result)

    # def execute_script(self, command: str, df: bool = False):
    #     with self.eng.begin() as conn: result = conn.executescript(text(command))
    #     return result if not df else pd.DataFrame(result)

    # ========================== TABLES =====================================
    def read_table(self, table: Optional[str] = None, sch: Optional[str] = None, size: int = 5):
        sch = sch or self.sch or 'main'
        if table is None:
            table = self.sch_tab[sch][0]
            print(f"Reading table `{table}` from schema `{sch}`")
        if self.con:
            res = self.con.execute(text(f'''SELECT * FROM {self._get_table_identifier(table, sch)} '''))
            return pd.DataFrame(res.fetchmany(size))

    def insert_dicts(self, table: str, *mydicts: dict[str, Any]) -> None:
        cmd = f"""INSERT INTO {table} VALUES """
        for mydict in mydicts: cmd += f"""({tuple(mydict)}), """
        self.execute_begin_once(cmd)

    def describe_table(self, table: str, sch: Optional[str] = None, dtype: bool = True) -> None:
        print(table.center(100, "="))
        self.refresh()
        assert self.meta is not None
        tbl = self.meta.tables[table]
        assert self.ses is not None
        count = self.ses.query(tbl).count()
        res = Struct(name=table, count=count, size_mb=count * len(tbl.exported_columns) * 10 / 1e6)
        res.print(dtype=False, as_config=True, title="TABLE DETAILS")
        dat = self.read_table(table=table, sch=sch, size=2)
        cols = self.get_columns(table, sch=sch)
        df = pd.DataFrame.from_records(dat, columns=cols)
        print("SAMPLE:\n", df)
        assert self.insp is not None
        if dtype: print("\nDETAILED COLUMNS:\n", pd.DataFrame(self.insp.get_columns(table)))
        print("\n" * 3)


if __name__ == '__main__':
    pass
