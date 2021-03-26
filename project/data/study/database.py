import sqlite3
import json
from ..data_file_path import COLOR_DB_PATH
from ...utils.tools import list_to_sql


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

    def write_trained_agent(self, item):
        self.cursor = self.con.cursor()
        request = """
        SELECT MAX(ID) FROM TrainedAgent
        """
        max_id = self.cursor.execute(request).fetchall()[0][0]
        if max_id is not None:
            max_id += 1
        else:
            max_id = 1
        item = max_id, *item
        request = """
                INSERT INTO TrainedAgent (id, hyper_param_id, training_data_id, accuracy, corpus_bleu, 
                training_accuracy, vocab_size, time_to_train)
                VALUES (?,?,?,?,?,?,?,?)
                """
        self.cursor.execute(request, item)
        self.con.commit()
        self.cursor.close()
        return max_id

    def read_hyper_parameters(self, model_id):
        self.cursor = self.con.cursor()
        request = """
        SELECT * FROM HyperParameters WHERE id IS {}""".format(model_id)
        self.cursor.execute(request)
        sql_output = self.cursor.fetchall()
        columns = self.get_column_names('HyperParameters')  # list of HyperParameters column names
        dict_data = {col: value for (col, value) in zip(columns, sql_output[0])}
        return dict_data

    def read_trained_agent(self, trained_agent_id):
        self.cursor = self.con.cursor()
        request = """
                SELECT * FROM TrainedAgent WHERE id IS {}""".format(trained_agent_id)
        self.cursor.execute(request)
        sql_output = self.cursor.fetchall()
        columns = self.get_column_names('TrainedAgent')  # list of TrainedAgent column names
        dict_data = {col: value for (col, value) in zip(columns, sql_output[0])}
        return dict_data

    def get_column_names(self, table_name):
        """"return a list of the column names of table table_name"""
        self.cursor = self.con.cursor()
        request = "SELECT * FROM {}".format(list_to_sql(table_name))
        self.cursor.execute(request)
        names = [description[0] for description in self.cursor.description]
        self.cursor.close()
        return names


