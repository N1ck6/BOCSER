import sqlite3
import os
from typing import List
import logging
logger = logging.getLogger(__name__)

class Connector:
    
    def __init__(
        self
    ) -> None:
        pass

    def set_request(
        self,
        request : str
    ) -> None:
        pass

    def get_request(
        self,
        request : str
    ) -> List:
        pass

class LocalConnector(Connector):

    def __init__(
        self, 
        db_filename : str = 'dihedral_logs.db'
    ) -> None:
        self.db_filename = db_filename
        if not os.path.isfile(db_filename):
            logger.error("No database file located: %s", db_filename)
            raise FileNotFoundError(db_filename)

    def set_request(
        self,
        request : str
    ) -> None:
        connection = sqlite3.connect(self.db_filename)
        try:
            cursor = connection.cursor()
            cursor.execute(request)
            connection.commit()
        except Exception as e:
            logger.exception("Something went wrong with db")
            raise e
        finally:
            connection.close()

    def get_request(
        self,
        request : str
    ) -> List:
        connection = sqlite3.connect(self.db_filename)
        try:
            cursor = connection.cursor()
            result = cursor.execute(request).fetchall()
            connection.commit()
            return result
        except Exception as e:
            logger.exception("Something went wrong with db")
            raise e
        finally:
            connection.close()

