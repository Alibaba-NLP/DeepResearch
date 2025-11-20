import os, pymysql
from dotenv import load_dotenv


MYSQL_HOST=os.getenv("MYSQL_HOST")
MYSQL_USER=os.getenv("MYSQL_USER")
MYSQL_PASSWORD=os.getenv("MYSQL_PASSWORD")
MYSQL_DB=os.getenv("MYSQL_DB")


# print(MYSQL_HOST)
# print(MYSQL_USER)
# print(MYSQL_PASSWORD)
# print(MYSQL_DB)

def _conn(db=None):
    return pymysql.connect(host=MYSQL_HOST,user=MYSQL_USER,password=MYSQL_PASSWORD,
                           database=db,charset="utf8mb4",autocommit=True)

def reset_docs():
    setup()
    with _conn(MYSQL_DB) as c:
        with c.cursor() as cur:
            cur.execute("TRUNCATE TABLE docs;")

def setup():
    with _conn() as c:
        with c.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    with _conn(MYSQL_DB) as c:
        with c.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS docs(
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                content LONGTEXT NOT NULL
            ) ENGINE=InnoDB;""")

def save_document(content: str) -> int:
    setup()
    with _conn(MYSQL_DB) as c:
        with c.cursor() as cur:
            cur.execute("INSERT INTO docs(content) VALUES (%s)", (content,))
            return cur.lastrowid

def fetch_document(doc_id: int) -> str|None:
    with _conn(MYSQL_DB) as c:
        with c.cursor() as cur:
            cur.execute("SELECT content FROM docs WHERE id=%s", (doc_id,))
            r = cur.fetchone()
            return r[0] if r else None

