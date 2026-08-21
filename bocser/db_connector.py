import sqlite3
import os
from typing import List, Optional
import hashlib
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

    def ensure_norm_energy_table(self) -> None:
        self.set_request(
            """CREATE TABLE IF NOT EXISTS norm_energies (
                   mol_hash TEXT NOT NULL,
                   theory_level TEXT NOT NULL,
                   norm_energy REAL NOT NULL,
                   source_mol_file TEXT,
                   created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                   PRIMARY KEY (mol_hash, theory_level)
               )"""
        )

    def get_norm_energy(self, mol_hash: str, theory_level: str) -> Optional[float]:
        self.ensure_norm_energy_table()
        rows = self.get_request(
            f"SELECT norm_energy FROM norm_energies "
            f"WHERE mol_hash = \"{mol_hash}\" AND theory_level = \"{theory_level}\""
        )
        return rows[0][0] if rows else None

    def set_norm_energy(self, mol_hash: str, theory_level: str, value: float, source_mol_file: str = "") -> None:
        self.ensure_norm_energy_table()
        self.set_request(
            "INSERT OR REPLACE INTO norm_energies (mol_hash, theory_level, norm_energy, source_mol_file) "
            f"VALUES (\"{mol_hash}\", \"{theory_level}\", {value!r}, \"{source_mol_file}\")"
        )

