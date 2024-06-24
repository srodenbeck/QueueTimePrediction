# -*- coding: utf-8 -*-

import read_db

if __name__=="__main__":
    print("Connecting to engine")
    engine = read_db.create_engine()
    print("Reading in dataframes")
    df_new_jobs_all = read_db.read_to_df("new_jobs_all", read_all=True)
    df_queue_priority = read_db.read_to_df("jobs_queue_priority_only", read_all=True)
    print("Merging dataframes")
    df_queue_priority.update(df_new_jobs_all)
    print("Uploading to database")
    df_queue_priority.to_sql('jobs_all', engine, if_exists='replace', index=False)
    print("Finished")
    
    
    
    
    
    