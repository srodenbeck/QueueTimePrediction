## List of features used for model
#### TODO: Put in alphabetical order once finished

- priority - Integer indicating slurm queue priority. Higher indicates job has greater precedence.
- req_cpus
- req_mem
- req_mem
- count_ahead
- work_ahead
- time_limit_raw

## Potential Features
Occupied_nodes = num nodes currently being used

remaining_cpu_time = sum of (reqCPUS * (requested time - elapsed time)) for all currently running jobs
	Can potentially include remaining cpu time for different factors, such as all currently running jobs with lower request size

Priority_position = rank of priority of requested job compared to other pending jobs (ex. highest priority would result in priority_position of 0, second highest would be 1, etc)

N_jobs_running = number of currently running jobs on the system


Queue_cpu_time = sum of (reqCPUS * (requested time - elapsed time)) for all jobs in queue
	Can potentially filter to just be jobs with higher priority than given job

Queue_req_nodes = sum of req_nodes for all jobs in queue

Queue_node_wall = sum of (req_nodes * (requested time - elapsed time)) for all jobs in queue

N_jobs_in_queue = number of jobs in queue

N_jobs_ahead = number of jobs presently in the queue with a higher priority than the given job

