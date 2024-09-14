
"""
https://docs.sqlalchemy.org/en/14/tutorial/index.html#a-note-on-the-future

"""

import time
from typing import Optional, Any, Callable

import pandas as pd

from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, text, inspect as inspect__, Engine, Connection
from sqlalchemy.engine import Inspector
from sqlalchemy.sql.schema import MetaData
from crocodile.core import Struct
from crocodile.file_management import List as L, P, OPLike


class DBMS:
    """Implementation Philosophy:
    * Always use sqlalchemy API and avoid sql-dielect specific language.
    * Engine is provided externally. It is the end-user's business to make this engine.
    """
    def __init__(self, engine: Engine, sch: Optional[str] = None, vws: bool = False, inspect: bool = True):
        self.eng: Engine = engine
        self.con: Optional[Connection] = None
        self.ses: Optional[Session] = None
        self.insp: Optional[Inspector] = None
        self.meta: Optional[MetaData] = None
        if self.eng.url.database and P(self.eng.url.database).exists():
            self.path: Optional[P] = P(self.eng.url.database)
        else: self.path = None

        # self.db = db
        self.sch = sch
        self.vws: bool = vws
        self.schema: list[str] = []
        # self.tables = None
        # self.views = None
        # self.sch_tab: Optional[Struct] = None
        # self.sch_vws: Optional[Struct] = None
        if inspect: self.refresh()
        # self.ip_formatter: Optional[Any] = None
        # self.db_specs: Optional[Any] = None
        if self.path is not None:
            if self.path.is_file(): path_repr = self.path.as_uri()
            else: path_repr = self.path
            print(f"Database at {path_repr} is ready.")

    def refresh(self, sch: Optional[str] = None) -> 'DBMS':  # fails if multiple schemas are there and None is specified
        self.con = self.eng.connect()
        self.ses = sessionmaker()(bind=self.eng)  # ORM style
        self.meta = MetaData()
        self.meta.reflect(bind=self.eng, schema=sch or self.sch)
        insp = inspect__(subject=self.eng)
        self.insp = insp
        self.schema = self.insp.get_schema_names()
        print(f"Inspecting tables of schema `{self.schema}` {self.eng}")
        self.sch_tab: dict[str, list[str]] = {k: v for k, v in zip(self.schema, [insp.get_table_names(schema=x) for x in self.schema])}  # dict(zip(self.schema, self.schema.apply(lambda x: self.insp.get_table_names(schema=x))))  #
        print(f"Inspecting views of schema `{self.schema}` {self.eng}")
        self.sch_vws: dict[str, list[str]] = {k: v for k, v in zip(self.schema, [insp.get_view_names(schema=x) for x in self.schema])}
        return self

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["con"]
        del state["ses"]
        del state["meta"]
        del state["insp"]
        del state["eng"]
        if self.path:
            state['path'] = self.path.collapseuser(strict=False)
        return state
    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.eng = self.make_sql_engine(self.path)
        self.refresh()

    @classmethod
    def from_local_db(cls, path: OPLike = None, echo: bool = False, share_across_threads: bool = False, pool_size: int = 5, **kwargs: Any):
        return cls(engine=cls.make_sql_engine(path=path, echo=echo, share_across_threads=share_across_threads, pool_size=pool_size, **kwargs))
    def __repr__(self): return f"DataBase @ {self.eng}"
    def get_columns(self, table: str, sch: Optional[str] = None):
        assert self.meta is not None
        return self.meta.tables[self._get_table_identifier(table=table, sch=sch)].exported_columns.keys()
    def close(self, sleep: int = 2):
        if self.path:
            print(f"Terminating database `{self.path.as_uri() if self.path.is_file() and 'memory' not in self.path else self.path}`")
        if self.con: self.con.close()
        if self.ses: self.ses.close()
        self.eng.pool.dispose()
        self.eng.dispose()
        time.sleep(sleep)
    def _get_table_identifier(self, table: str, sch: Optional[str]):
        if sch is None: sch = self.sch
        if sch is not None:
            return f"""{sch}."{table}" """
        else: return table

    @staticmethod
    def make_sql_engine(path: OPLike = None, echo: bool = False, dialect: str = "sqlite", driver: str = ["pysqlite", "DBAPI"][0], pool_size: int = 5, share_across_threads: bool = True, **kwargs: Any):
        """Establish lazy initialization with database"""
        from sqlalchemy.pool import StaticPool, NullPool
        _ = NullPool
        if str(path) == "memory":
            print("Linking to in-memory database.")
            if share_across_threads:
                # see: https://docs.sqlalchemy.org/en/14/dialects/sqlite.html#using-a-memory-database-in-multiple-threads
                return create_engine(url=f"{dialect}+{driver}:///:memory:", echo=echo, future=True, poolclass=StaticPool, connect_args={"check_same_thread": False})
            else:
                return create_engine(url=f"{dialect}+{driver}:///:memory:", echo=echo, future=True, pool_size=pool_size, **kwargs)
        path = P.tmpfile(folder="tmp_dbs", suffix=".sqlite") if path is None else P(path).expanduser().absolute().create(parents_only=True)
        path_repr = path.as_uri() if path.is_file() else path
        print(f"Linking to database at {path_repr}")
        if pool_size == 0:
            res = create_engine(url=f"{dialect}+{driver}:///{path}", echo=echo, future=True, poolclass=NullPool, **kwargs)  # echo flag is just a short for the more formal way of logging sql commands.
        else:
            res = create_engine(url=f"{dialect}+{driver}:///{path}", echo=echo, future=True, pool_size=pool_size, **kwargs)  # echo flag is just a short for the more formal way of logging sql commands.
        return res

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

    def execute(self, command: str):
        with self.eng.begin() as conn:
            result = conn.execute(text(command))
        return result

    # def execute_script(self, command: str, df: bool = False):
    #     with self.eng.begin() as conn: result = conn.executescript(text(command))
    #     return result if not df else pd.DataFrame(result)

    # ========================== TABLES =====================================
    def read_table(self, table: Optional[str] = None, sch: Optional[str] = None, size: int = 5):
        if sch is None:
            schemas = [a_sch for a_sch in self.schema if a_sch not in ["information_schema", "pg_catalog"]]
            if len(schemas) > 1 and "public" in schemas:
                schemas.remove("public")
        if table is None:
            tables = self.sch_tab[sch]
            assert len(tables) > 0, f"No tables found in schema `{sch}`"
            table = L(tables).sample(size=1).list[0]
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
