"""
Database Connection Manager

Handles database connections, sessions, and operations.
"""

import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

from data.storage.models import Base

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and sessions.

    Singleton pattern to ensure single database connection pool.
    """

    _instance = None
    _engine = None
    _session_factory = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._engine is None:
            self.initialize()

    def initialize(self, database_url: str = None):
        """
        Initialize database connection.

        Args:
            database_url: Database URL. If None, read from env
        """
        if database_url is None:
            database_url = os.getenv('DATABASE_URL', 'sqlite:///data/trading_signals.db')

        logger.info(f"Initializing database connection...")

        # Create engine
        if database_url.startswith('sqlite'):
            # SQLite configuration
            self._engine = create_engine(
                database_url,
                echo=False,
                connect_args={'check_same_thread': False}
            )
        else:
            # PostgreSQL configuration
            self._engine = create_engine(
                database_url,
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True  # Verify connections before using
            )

        # Create session factory
        self._session_factory = scoped_session(
            sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False
            )
        )

        logger.info("Database connection initialized")

    @property
    def engine(self):
        """Get database engine"""
        if self._engine is None:
            self.initialize()
        return self._engine

    @property
    def session_factory(self):
        """Get session factory"""
        if self._session_factory is None:
            self.initialize()
        return self._session_factory

    def create_tables(self):
        """Create all database tables"""
        logger.info("Creating database tables...")
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created")

    def drop_tables(self):
        """Drop all database tables (USE WITH CAUTION!)"""
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(self.engine)
        logger.info("Database tables dropped")

    @contextmanager
    def get_session(self):
        """
        Get a database session (context manager).

        Usage:
            with db_manager.get_session() as session:
                session.add(obj)
                session.commit()
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def close(self):
        """Close database connections"""
        if self._session_factory:
            self._session_factory.remove()
        if self._engine:
            self._engine.dispose()
        logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
def get_session():
    """Get a database session (context manager)"""
    return db_manager.get_session()


def init_database(database_url: str = None):
    """
    Initialize database and create tables.

    Args:
        database_url: Database URL. If None, read from env
    """
    db_manager.initialize(database_url)
    db_manager.create_tables()
