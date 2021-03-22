import dbutils

def create_tables():

	pan_sql = """
	CREATE TABLE IF NOT EXISTS pan (
	    id text PRIMARY KEY,
	    name text NOT NULL,
	    fathers_name text,
	    dob text NOT NULL)"""
	    
	dbutils.run_query(pan_sql)

if __name__ == '__main__':

	create_tables()