## Work in progress

## Columns
- job_id (PK INT) - Job ID 
- user_id (INT) - User ID 
- account (TEXT) - Account name 
- state (STATE_ENUM) - State of job. 
- partition (PARTITION_ENUM). Partition that job ran in. Options not in ENUM are deleted from table
- time_limit_raw (INT) - Time limit for job in minutes
- submit (TIMESTAMP) - Time job was submitted.
- eligible (TIMESTAMP) - Time job became eligible to run.
- elapsed (INT) - Elapsed time of job in seconds.
- planned (INT) - Time job spent waiting between eligible to start. Saved in seconds (also called queue_time).
- start_time (TIMESTAMP) - Time job was started.
- end_time (TIMESTAMP) - Time job ended.
- priority (INT) - Priority given to job by SLURM. Higher is given more precedence.
- req_cpus (INT) - Number of CPUS requested for job.
- req_mem (REAL) - Amount of memory requested in Gigabytes.
- req_nodes (INT) - Number of nodes requested.
- req_tres (TEXT) - Text string represented requested resources.
- qos (TEXT) - Quality of Service for job. 

### ENUMS
- STATE_ENUM = ['COMPLETED', 'CANCELLED', 'FAILED', 'REQUEUED', 'NODE_FAIL', 'PENDING', 
            'OUT_OF_MEMORY', 'TIMEOUT']
- PARTITION_ENUM = ['standard', 'shared', 'wholenode', 'wide', 'gpu', 'highmem', 
    'azure']