import sqlite3
import json
from ..data_file_path import COLOR_DB_PATH


class ColorDB:
    """
    Object used to connect to ColorDB database, write and get data from it. This is the database were all the
    results are saved.
    The connection to the database is done once on the instance creation and then the instance can be used for multiple
    requests. The connection is closed on instance destruction."""

    def __init__(self):
        """
        on instance creation, use connection con_sqlite to a  sqlite database and creates a cursor
         """

        self.con = sqlite3.connect(COLOR_DB_PATH, timeout=120)
        # initially the same cursor was used and left opened during program execution. It is bad practice a it can lead
        # to database is locked issues and it is better to open a cursor and close it just afterwards
        self.cursor = None

    def write_hyper_search(self, item):
        """
        :param item: list with the data to save
        """
        self.cursor = self.con.cursor()
        for k in [1, 2, 3]:
            item[k] = json.dumps(item[k])  # turns dict to a string
        request = """
        INSERT INTO HyperSearch (model, fixed_params, param_grid, best_params, best_score)
        VALUES (?,?,?,?,?)
        """
        self.cursor.execute(request, item)
        self.con.commit()
        self.cursor.close()
