
"""DB
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import psutil

engine = create_engine('sqlite:///:memory:')


def generate_random_dataframe():
    num_rows = 1000000
    num_columns = 10
    column_names: list[str] = [''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 5)) for _ in range(num_columns)]
    data = np.random.rand(num_rows, num_columns)
    df_ = pd.DataFrame(data, columns=column_names)
    return df_


for i in range(10):
    df = generate_random_dataframe()
    table_name = 'table_' + str(i)
    df.to_sql(table_name, engine, index=False)


memory_before = psutil.virtual_memory().used / (1024 ** 3)  # Memory usage in GB
print(f"Memory Usage Before Dropping Tables: {memory_before:.2f} GB")

# Loop over the tables, drop them, and check memory usage after each drop
for i in range(10):
    table_name = 'table_' + str(i)
    connection = engine.connect()
    connection.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
    connection.close()

memory_final = psutil.virtual_memory().used / (1024 ** 3)  # Memory usage in GB
print(f"Memory Usage After Dropping All Tables: {memory_final:.2f} GB")

engine.pool.dispose()
memory_final = psutil.virtual_memory().used / (1024 ** 3)  # Memory usage in GB
print(f"Memory Usage After Disposing Connections: {memory_final:.2f} GB")
