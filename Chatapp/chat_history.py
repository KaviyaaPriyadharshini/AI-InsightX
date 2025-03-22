import sqlite3
import streamlit as st

def init_db():
    conn = sqlite3.connect("chat_history.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS chat_history")
    conn.commit()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            question TEXT,
            answer TEXT,
            UNIQUE(source, question) 
        )
    """)
    conn.commit()
    return conn, cursor

conn, cursor = init_db()
def save_chat_history(source, question, answer):
    try:
        cursor.execute(
            "INSERT INTO chat_history (source, question, answer) VALUES (?, ?, ?)",
            (source, question, answer),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass  
    
def load_chat_history():
    cursor.execute("SELECT source, question, answer FROM chat_history ORDER BY source")
    return cursor.fetchall()
