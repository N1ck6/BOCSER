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
            # Create an empty SQLite file and continue
            parent = os.path.dirname(os.path.abspath(db_filename))
            if parent:
                os.makedirs(parent, exist_ok=True)
            open(db_filename, "a").close()
            logger.info(
                "No database file at %s — created empty dihedral_logs.db and needed tables.",
                db_filename,
            )
        self.ensure_all_tables()

    def ensure_all_tables(self) -> None:
        """Create all cache tables if they do not exist (idempotent)."""
        self.ensure_dihedrals_table()
        self.ensure_norm_energy_table()
        self.ensure_ts_bond_coefs_table()

    def ensure_dihedrals_table(self) -> None:
        """Fourier torsion-scan coefficients (fragment SMILES + method)."""
        self.set_request(
            """CREATE TABLE IF NOT EXISTS dihedrals (
                   dihedral_smiles TEXT NOT NULL,
                   method TEXT NOT NULL,
                   a1 REAL, a2 REAL, a3 REAL,
                   b1 REAL, b2 REAL, b3 REAL,
                   c REAL,
                   PRIMARY KEY (dihedral_smiles, method)
               )"""
        )

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
        rows = self.get_request_params(
            "SELECT norm_energy FROM norm_energies WHERE mol_hash = ? AND theory_level = ?",
            (mol_hash, theory_level),
        )
        return rows[0][0] if rows else None

    def set_norm_energy(self, mol_hash: str, theory_level: str, value: float, source_mol_file: str = "") -> None:
        self.ensure_norm_energy_table()
        self.set_request_params(
            "INSERT OR REPLACE INTO norm_energies (mol_hash, theory_level, norm_energy, source_mol_file) "
            "VALUES (?, ?, ?, ?)",
            (mol_hash, theory_level, value, source_mol_file),
        )

    def set_request_params(self, request: str, params: tuple) -> None:
        """Parameterized (?, ?, ...) — safe against special characters in SMILES/strings."""
        connection = sqlite3.connect(self.db_filename)
        try:
            cursor = connection.cursor()
            cursor.execute(request, params)
            connection.commit()
        except Exception as e:
            logger.exception("Something went wrong with db (parameterized)")
            raise e
        finally:
            connection.close()

    def get_request_params(self, request: str, params: tuple) -> List:
        connection = sqlite3.connect(self.db_filename)
        try:
            cursor = connection.cursor()
            result = cursor.execute(request, params).fetchall()
            connection.commit()
            return result
        except Exception as e:
            logger.exception("Something went wrong with db (parameterized)")
            raise e
        finally:
            connection.close()

    def ensure_ts_bond_coefs_table(self) -> None:
        self.set_request(
            """CREATE TABLE IF NOT EXISTS ts_bond_coefs (
                   mol_hash TEXT NOT NULL,
                   theory_level TEXT NOT NULL,
                   atom_pair_smiles TEXT NOT NULL,
                   de REAL NOT NULL, a REAL NOT NULL, re REAL NOT NULL, c REAL NOT NULL,
                   PRIMARY KEY (mol_hash, theory_level, atom_pair_smiles)
               )"""
        )

    def get_ts_bond_coefs(self, mol_hash, theory_level, atom_pair_smiles):
        self.ensure_ts_bond_coefs_table()
        rows = self.get_request_params(
            "SELECT de, a, re, c FROM ts_bond_coefs "
            "WHERE mol_hash = ? AND theory_level = ? AND atom_pair_smiles = ?",
            (mol_hash, theory_level, atom_pair_smiles),
        )
        return rows[0] if rows else None

    def set_ts_bond_coefs(self, mol_hash, theory_level, atom_pair_smiles, coefs):
        self.ensure_ts_bond_coefs_table()
        self.set_request_params(
            "INSERT OR REPLACE INTO ts_bond_coefs "
            "(mol_hash, theory_level, atom_pair_smiles, de, a, re, c) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mol_hash, theory_level, atom_pair_smiles, *[float(v) for v in coefs]),
        )
