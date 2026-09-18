import json, sqlite3
from datetime import datetime, timezone, timedelta
from threading import Lock
from app.config import settings
from app.models.schemas import Product, Event

class Store:
    def __init__(self, db_path: str | None = None):
        path = db_path or settings.database_path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = Lock()
        self.version = 0
        self._init()

    def _init(self):
        with self.lock, self.db:
            self.db.execute("PRAGMA journal_mode=WAL;")
            self.db.execute("PRAGMA synchronous=NORMAL;")
            self.db.executescript('''
            CREATE TABLE IF NOT EXISTS products (id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL,
                user_id TEXT, anonymous_id TEXT, session_id TEXT, product_id TEXT,
                query TEXT, category TEXT, timestamp TEXT NOT NULL, location TEXT, metadata TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id, anonymous_id);
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_product ON events(product_id);
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
            ''')

    def upsert_product(self, p: Product):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO products(id,data) VALUES(?,?)", (p.id, p.model_dump_json()))
            self.version += 1

    def upsert_products(self, products: list[Product]):
        with self.lock, self.db:
            self.db.executemany("INSERT OR REPLACE INTO products(id,data) VALUES(?,?)", [(p.id, p.model_dump_json()) for p in products])
            self.version += 1

    def products(self) -> list[Product]:
        with self.lock:
            return [Product.model_validate_json(r["data"]) for r in self.db.execute("SELECT data FROM products")]

    def product(self, pid: str) -> Product | None:
        with self.lock:
            r = self.db.execute("SELECT data FROM products WHERE id=?", (pid,)).fetchone()
            return Product.model_validate_json(r["data"]) if r else None

    def add_event(self, e: Event):
        with self.lock, self.db:
            self.db.execute("""INSERT INTO events(event,user_id,anonymous_id,session_id,product_id,query,category,timestamp,location,metadata)
                VALUES(?,?,?,?,?,?,?,?,?,?)""", (
                e.event,
                e.user_id,
                e.anonymous_id,
                e.session_id,
                e.product_id,
                e.query,
                e.category,
                e.timestamp.isoformat(),
                json.dumps(e.location.model_dump()) if e.location else None,
                json.dumps(e.metadata)
            ))
            self.version += 1

    def events(self, user_id=None, anonymous_id=None, session_id=None, limit=1000):
        clauses = []
        args = []
        if user_id:
            clauses.append("user_id=?")
            args.append(user_id)
        elif anonymous_id:
            clauses.append("anonymous_id=?")
            args.append(anonymous_id)
        if session_id:
            clauses.append("session_id=?")
            args.append(session_id)

        # Note: clauses are strictly constructed from static column names above,
        # safe against SQL injection; arguments are parameterized.
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.lock:
            rows = self.db.execute(f"SELECT * FROM events{where} ORDER BY timestamp DESC LIMIT ?", (*args, limit)).fetchall()
            return rows

    def all_events(self, limit=100000, since_days: int | None = None):
        with self.lock:
            if since_days is not None:
                cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
                return self.db.execute(
                    "SELECT * FROM events WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                    (cutoff, limit)
                ).fetchall()
            return self.db.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()

    def close(self):
        with self.lock:
            self.db.close()

store = Store()
