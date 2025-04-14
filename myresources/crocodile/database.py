
import time
from typing import Optional, Any, Callable

import pandas as pd

from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, text, inspect as inspect__, Engine, Connection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.engine import Inspector
from sqlalchemy.sql.schema import MetaData
from crocodile.core import Struct, List as L
from crocodile.file_management import P, OPLike


class DBMS:
    def __init__(self, engine: Engine, sch: Optional[str] = None, vws: bool = False):
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

        self.sch_tab: dict[str, list[str]]
        self.sch_vws: dict[str, list[str]]
        # self.tables = None
        # self.views = None
        # self.sch_tab: Optional[Struct] = None
        # self.sch_vws: Optional[Struct] = None
        # if inspect: self.refresh()
        # self.ip_formatter: Optional[Any] = None
        # self.db_specs: Optional[Any] = None
        if self.path is not None:
            if self.path.is_file():
                path_repr = self.path.as_uri()
            else:
                path_repr = self.path
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
        self.sch_tab = {k: v for k, v in zip(self.schema, [insp.get_table_names(schema=x) for x in self.schema])}  # dict(zip(self.schema, self.schema.apply(lambda x: self.insp.get_table_names(schema=x))))  #
        print(f"Inspecting views of schema `{self.schema}` {self.eng}")
        self.sch_vws = {k: v for k, v in zip(self.schema, [insp.get_view_names(schema=x) for x in self.schema])}
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
            return f""" "{sch}"."{table}" """
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

    @staticmethod
    def make_sql_async_engine(path: OPLike = None, echo: bool = False, dialect: str = "sqlite", driver: str = "aiosqlite", pool_size: int = 5, share_across_threads: bool = True, **kwargs: Any):
        """Establish lazy initialization with database"""
        from sqlalchemy.pool import StaticPool, NullPool
        _ = NullPool
        if str(path) == "memory":
            print("Linking to in-memory database.")
            if share_across_threads:
                # see: https://docs.sqlalchemy.org/en/14/dialects/sqlite.html#using-a-memory-database-in-multiple-threads
                return create_async_engine(url=f"{dialect}+{driver}://", echo=echo, future=True, poolclass=StaticPool, connect_args={"mode": "memory", "cache": "shared"})
            else:
                return create_async_engine(url=f"{dialect}+{driver}:///:memory:", echo=echo, future=True, pool_size=pool_size, **kwargs)
        path = P.tmpfile(folder="tmp_dbs", suffix=".sqlite") if path is None else P(path).expanduser().absolute().create(parents_only=True)
        path_repr = path.as_uri() if path.is_file() else path
        print(f"Linking to database at {path_repr}")
        if pool_size == 0:
            res = create_async_engine(url=f"{dialect}+{driver}:///{path}", echo=echo, future=True, poolclass=NullPool, **kwargs)  # echo flag is just a short for the more formal way of logging sql commands.
        else:
            res = create_async_engine(url=f"{dialect}+{driver}:///{path}", echo=echo, future=True, pool_size=pool_size, **kwargs)  # echo flag is just a short for the more formal way of logging sql commands.
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
            conn.commit()
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
            sch = schemas[0]
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

    def describe_db(self):
        self.refresh()
        assert self.meta is not None
        res_all = []
        assert self.ses is not None
        from tqdm import tqdm
        for tbl in tqdm(self.meta.sorted_tables, desc="Inspecting tables", unit="table"):
            table = tbl.name
            if self.sch is not None:
                table = f"{self.sch}.{table}"
            count = self.ses.query(tbl).count()
            res = dict(table=table, count=count, size_mb=count * len(tbl.exported_columns) * 10 / 1e6,
                       columns=len(tbl.exported_columns), schema=self.sch)
            res_all.append(res)
        return pd.DataFrame(res_all)

    def describe_table(self, table: str, sch: Optional[str] = None, dtype: bool = True) -> None:
        print(table.center(100, "="))
        self.refresh()
        assert self.meta is not None
        tbl = self.meta.tables[table]
        assert self.ses is not None
        count = self.ses.query(tbl).count()
        res = Struct(dict(name=table, count=count, size_mb=count * len(tbl.exported_columns) * 10 / 1e6))
        res.print(dtype=False, as_config=True, title="TABLE DETAILS")
        dat = self.read_table(table=table, sch=sch, size=2)
        cols = self.get_columns(table, sch=sch)
        df = pd.DataFrame.from_records(dat, columns=cols)
        print("SAMPLE:\n", df)
        assert self.insp is not None
        if dtype: print("\nDETAILED COLUMNS:\n", pd.DataFrame(self.insp.get_columns(table)))
        print("\n" * 3)


DB_TMP_PATH = P.tmp().joinpath("tmp_dbs/results/data.sqlite")


def to_db(table: str, idx: int, idx_max: int, data: Any):
    import pickle
    db = DBMS.from_local_db(DB_TMP_PATH)
    time_now = time.time_ns()
    data_blob = pickle.dumps(data)
    create_table = f"""CREATE TABLE IF NOT EXISTS "{table}" (time INT PRIMARY KEY, idx INT, idx_max INT, data BLOB)"""
    insert_row = f"""INSERT INTO "{table}" (time, idx, idx_max, data) VALUES (:time, :idx, :idx_max, :data)"""
    with db.eng.connect() as conn:
        conn.execute(text(create_table))
        conn.execute(
            text(insert_row),
            {'time': time_now, 'idx': idx, 'idx_max': idx_max, 'data': data_blob}
        )
        conn.commit()
    db.close()


def from_db(table: str):
    import pickle
    db = DBMS.from_local_db(DB_TMP_PATH)
    with db.eng.connect() as conn:
        res = conn.execute(text(f"""SELECT * FROM "{table}" """))
        records = res.fetchall()
        df = pd.DataFrame(records, columns=['time', 'idx', 'idx_max', 'data'])
        df['data'] = df['data'].apply(pickle.loads)
        return df


def get_table_specs(engine: Engine, table_name: str) -> pd.DataFrame:
    inspector = inspect__(engine)
    # Collect table information
    columns_info = [{
        'name': col['name'],
        'type': str(col['type']),
        'nullable': col['nullable'],
        'default': col['default'],
        'autoincrement': col.get('autoincrement'),
        'category': 'column'
    } for col in inspector.get_columns(table_name)]
    # Primary keys
    pk_info = [{
        'name': pk,
        'type': None,
        'nullable': False,
        'default': None,
        'autoincrement': None,
        'category': 'primary_key'
    } for pk in inspector.get_pk_constraint(table_name)['constrained_columns']]
    # Foreign keys
    fk_info = [{
        'name': fk['constrained_columns'][0],
        'type': f"FK -> {fk['referred_table']}.{fk['referred_columns'][0]}",
        'nullable': None,
        'default': None,
        'autoincrement': None,
        'category': 'foreign_key'
    } for fk in inspector.get_foreign_keys(table_name)]
    # Indexe
    index_info = [{
        'name': idx['name'],
        'type': f"Index on {', '.join(idx['column_names'])}",
        'nullable': None,
        'default': None,
        'autoincrement': None,
        'category': 'index',
        'unique': idx['unique']
    } for idx in inspector.get_indexes(table_name)]
    # Combine all information
    all_info = columns_info + pk_info + fk_info + index_info
    # Convert to DataFrame
    df = pd.DataFrame(all_info)
    return df

if __name__ == '__main__':
    pass
