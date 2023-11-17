from ekonlpy.sentiment import MPCK
import sqlite3

table_name = "news_comment"

# Create a function to update sentiment values for a specific table
def main():
    print(f"Updating sentiment values for table: {table_name}")
    
    # Connect to the existing SQLite database
    conn_existing = sqlite3.connect("/Users/siwon/Desktop/Fall-23/research/official_news_dataset/news_dataset.sqlite")
    cursor_existing = conn_existing.cursor()

    # Execute an SQL query to retrieve the data from the specified table
    cursor_existing.execute(f'SELECT comment_id, text_display FROM {table_name}')
    comments = cursor_existing.fetchall()

    # Close the read-only database connection
    conn_existing.close()

    # Connect to the same database for updating
    conn_update = sqlite3.connect("/Users/siwon/Desktop/Fall-23/research/official_news_dataset/news_dataset.sqlite")
    cursor_update = conn_update.cursor()

    # Create an instance of the MPCK model
    mpck = MPCK()

    # Iterate through the retrieved data and update the specified table
    for comment_id, text_display in comments[440431:]:
        # Process the data
        tokens = mpck.tokenize(text_display)
        ngrams = mpck.ngramize(tokens)
        score = mpck.classify(tokens + ngrams, intensity_cutoff=1.5)
        print(f"Updating sentiment values for comment_id: {comment_id}")
        # Update the values in the specified table
        cursor_update.execute(f'UPDATE {table_name} SET polarity = ?, intensity = ?, pos_score = ?, neg_score = ? WHERE comment_id = ?',
                             (score['Polarity'], score['Intensity'], score['Pos score'], score['Neg score'], comment_id))

        # Commit the changes to the database
        conn_update.commit()

    # Close the database connections
    conn_update.close()
    
    print(f"Finished updating sentiment values for table: {table_name}")

if __name__ == "__main__":
    main()