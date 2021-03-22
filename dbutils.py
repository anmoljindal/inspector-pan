import os
import sqlite3

# create a default path to connect to and create (if necessary) a database
# called 'database.sqlite3' in the same directory as this script
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'database.sqlite3')

def db_connect(db_path=DEFAULT_PATH):
    con = sqlite3.connect(db_path)
    return con

def run_query(query):

	connection = db_connect() # connect to the database
	cursor = connection.cursor() # instantiate a cursor obj
	cursor.execute(query)
	connection.close()