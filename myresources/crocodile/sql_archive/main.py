
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import Column, String, Integer, Float, DateTime, create_engine
import crocodile.toolbox as tb


def get_engine_session(path=None, echo=False):
    if path is None: path = tb.P.tmp(file='first_database.db')
    engine = create_engine(url=f"sqlite:///{path}", echo=echo)
    sess = sessionmaker(bind=engine)
    return engine, sess


def create_table():
    Base = declarative_base()


    class UserTable(Base):
        __tablename__ = "users"
        id = Column(Integer(), primary_key=True)
        username = Column(String(length=25), nullable=False, unique=True)
        email = Column(String(85), unique=True, nullable=False)
        weight = Column(Float(), nullable=True)
        date_created = Column(DateTime(), default=tb.datetime.utcnow)

        def __repr__(self):
            return f"User {self.id=}"


def create_db(eng, Base):  # to be ran only once.
    Base.metadata.create_all(eng)


def update_table_attrs():
    Base = automap_base()


    class Users(Base):
        __tablename__ = 'users'
        # Override id column, the type must match. Automap handles the rest.
        id = Column(Integer, primary_key=True)

    # Continue with the automapping. Will fill in the rest.


    # Base.prepare(engine, reflect=True)


def interact(eng):
    sess = sessionmaker(bind=eng)
    # or conn = engine.connect() conn.add etc
    # new_user = UserTable(id=2, username="lol", email="ha@ha.com")
    # sess.add(new_user)
    # sess.commit()


if __name__ == '__main__':
    pass
