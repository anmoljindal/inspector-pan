import dbutils

def create_tables():

	connection = db_connect() # connect to the database
	cursor = connection.cursor() # instantiate a cursor obj
	pan_sql = """
	CREATE TABLE pan (
	    id text PRIMARY KEY,
	    name text NOT NULL,
	    fathers_name text,
	    dob text NOT NULL)"""
	cursor.execute(pan_sql)

if __name__ == '__main__':

	create_tables()