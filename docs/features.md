## List of features used for model
#### TODO: Put in alphabetical order once finished

# Target
- planned/queue_time (INT) - Time job spent waiting between eligible to start. Saved in seconds.

# Features
#### Straight from sacct
- partition (PARTITION_ENUM). Partition that job ran in. Options not in ENUM are deleted from table
- time_limit_raw (INT) - Time limit for job in minutes
- planned (INT) - start_time - eligible. Saved in seconds.
- priority (INT) - Priority given to job by SLURM. Higher is given more precedence.
- req_cpus (INT) - Number of CPUS requested for job.
- req_mem (REAL) - Amount of memory requested in Gb.
- req_nodes (INT) - Number of nodes requested.
- req_tres (TEXT) - Text string represented requested resources.
- qos (TEXT) - Quality of Service for job. 
#### Computation required
- jobs_ahead_queue (INT) - Number of other jobs ahead in queue. 
- time_limit_ahead_queue (INT) - Summed time limit of jobs ahead in queue.
- cpus_ahead_queue (INT) - Sum of CPUs requested by jobs ahead in queue.
- nodes_ahead_queue (INT) - Sum of nodes requested by jobs ahead in queue.
- memory_ahead_queue (REAL) - Sum of memory requested by jobs ahead in queue in Gb.
- jobs_running (INT) - Number of jobs currently running.
- cpus_running (INT) - Number of CPUs currently being used.
- nodes_running (INT) - Number of nodes currently being used.
- memory_running (REAL) - Amount of memory currently being used in Gb.

### Potential Features
- count_ahead_user
- work_ahead_user
- count_ahead_project
- work_ahead_project

remaining_cpu_time = sum of (reqCPUS * (requested time - elapsed time)) for all currently running jobs
	Can potentially include remaining cpu time for different factors, such as all currently running jobs with lower request size

Priority_position = rank of priority of requested job compared to other pending jobs (ex. highest priority would result in priority_position of 0, second highest would be 1, etc)

Queue_cpu_time = sum of (reqCPUS * (requested time - elapsed time)) for all jobs in queue
	Can potentially filter to just be jobs with higher priority than given job

Queue_node_wall = sum of (req_nodes * (requested time - elapsed time)) for all jobs in queue

N_jobs_ahead = number of jobs presently in the queue with a higher priority than the given job

