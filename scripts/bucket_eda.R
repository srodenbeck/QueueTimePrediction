library(RMariaDB)
library(RPostgres)
library(dplyr)

# Connect to databases
psql_con <- dbConnect(
  RPostgres::Postgres(),
  dbname = 'sacctdata',
  host = 'slurm-data-loadbalancer.reu-p4.anvilcloud.rcac.purdue.edu',
  port = 5432,
  user = 'postgres',
  password = 'stack2024'
)
maria_con <- dbConnect(RMariaDB::MariaDB(),
                       user = 'dbadmin',
                       # CHANGE TO PASSWORD WHEN RUNNING
                       password = 'HIDDEN :)',
                       dbname = 'glue2',
                       host = 'localhost',
                       port = 3306)

# Get datasets
maria_result <- dbGetQuery(maria_con, "SELECT * FROM jobs WHERE startTime IS NOT NULL")
psql_result <- dbGetQuery(psql_con, "SELECT * FROM new_jobs_all WHERE start_time IS NOT NULL")

# Make delay column
psql_delay <- psql_result$start_time - psql_result$submit + 1
maria_delay <- maria_result$startTime - maria_result$submitTime + 1
maria_result$delay <- maria_delay
psql_result$delay <- psql_delay

psql_result$delay <- as.integer(psql_result$delay) / 60
maria_result$delay <- as.integer(maria_result$delay) / 60

calculate_means <- function(df) {
  valid_values <- x[is.finite(x)]
  mean(valid_values, na.rm = TRUE)
}

psql_ten <- psql_result %>%
  filter(delay < 10)
psql_overfour <- psql_result %>%
  filter(delay > 4 * 60)

ten_means <- calculate_means(psql_ten)
overfour_means <- calculate_means(psql_overfour)
mean_diff <- overfour_means - ten_means

for (col_name in names(mean_diff)) {
  cat(col_name, ":", mean_diff[col_name], "\n")
}


mean(psql_overfour$time_limit_raw) - mean(psql_ten$time_limit_raw)
# Req_cpus isn't really helpful, difference of -31
mean(psql_overfour$req_cpus) - mean(psql_ten$req_cpus)
mean(psql_overfour$jobs_ahead_queue) - mean(psql_ten$jobs_ahead_queue)
mean(psql_overfour$jobs_running) - mean(psql_ten$jobs_running)
mean(psql_overfour$req_mem) - mean(psql_ten$req_mem)
mean(psql_overfour$req_nodes) - mean(psql_ten$req_nodes)
mean(psql_overfour$cpus_ahead_queue) - mean(psql_ten$cpus_ahead_queue)
mean(psql_overfour$priority) - mean(psql_ten$priority)


# Priority, 





