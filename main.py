import os
import sqlite3

DATABASE_PATH = os.path.join("sqlite", "dataset.sqlite")


def get_database_tables(db_connection: sqlite3.Connection):
    try:
        print("Connected to SQLite")

        # Getting all tables from sqlite_master
        sql_query = """SELECT name FROM sqlite_master 
        WHERE type='table';"""

        # Creating cursor object using connection object
        cursor = db_connection.cursor()

        # executing our sql query
        cursor.execute(sql_query)
        print("List of tables\n")

        # printing all tables list
        print(cursor.fetchall())

    except sqlite3.Error as error:
        print("Failed to execute the above query", error)


def main() -> None:
    print(f"Using database at: {DATABASE_PATH}")
    database_path = DATABASE_PATH + "?mode=ro"
    # Connect to the SQLite database in read-only mode
    conn = sqlite3.connect(database_path, uri=True)
    conn.execute("PRAGMA query_only = 1")
    get_database_tables(conn)
    conn.close()

    # try:
    #     # Execute a query to fetch all records from the 'samples' table
    #     df = pd.read_sql_query(
    #         """
    #         SELECT
    #           location_id,
    #           lat,
    #           lon,
    #           heading,
    #           capture_date,
    #           pano_id,
    #           batch_date,
    #           image
    #         FROM samples
    #         """,
    #         conn,
    #     )
    # finally:
    #     conn.close()

    # if df.empty:
    #     print("No data found in the 'samples' table.")
    #     return

    # print(f"Loaded {len(df)} records from the 'samples' table.")
    # print("Sample records:")
    # print(df.head())
    print("Program Done!")


# Start the server
if __name__ == "__main__":
    main()
