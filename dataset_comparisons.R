library(RMariaDB)
library(RPostgres)

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
                 password = 'REMOVED FOR PRIVACY REASONS',
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

# Null hypothesis: there is not a difference between psql_delay and maria_delay
# Alternative hypothesis: there is a difference betwen psql_delay and maria_delay
# Testing with alpha = 0.01
# psql_mean <- mean(psql_delay) yields 7167.159 secs
psql_mean <- 7167.159
psql_sd <- sd(psql_delay)
psql_n <- length(psql_delay)

# maria_mean <- mean(maria_delay) yields 24433.38 secs
maria_mean <- 24433.38
maria_sd <- sd(maria_delay)
maria_n <- length(maria_delay)

# Without assuming equal variances
tts <- (maria_mean - psql_mean) / sqrt(psql_sd * psql_sd / psql_n
                                     + maria_sd * maria_sd / maria_n)
df <- ((psql_sd * psql_sd / psql_n + maria_sd * maria_sd / maria_n) ^ 2) /
      ((psql_sd * psql_sd / psql_n) ^ 2 / (psql_n - 1) +
      (maria_sd * maria_sd / maria_n) ^ 2 / (maria_n - 1))

p_value <- 2 * pt(-abs(tts), df)
# p_value = 0
# Therefore we reject the null hypothesis that there is not a difference between
# maria_mean and psql_mean (as p-value of 0 is less than alpha value of 0.01)

# Graphing!
library(ggplot2)

n_bins <- 30

maria_result$log_delay <- log(as.numeric(maria_result$delay))
psql_result$log_delay <- log(as.numeric(psql_result$delay))
maria_log_delay <- na.omit(maria_result$log_delay)
psql_log_delay <- na.omit(psql_result$log_delay)
maria_log_delay <- maria_log_delay[is.finite(maria_log_delay)]
psql_log_delay <- psql_log_delay[is.finite(psql_log_delay)]

# ___1 is postgres, ___2 is mariadb
xbar1 <- mean(psql_log_delay)
s1 <- sd(psql_log_delay)
n1 <- length(psql_log_delay)
xbar2 <- mean(maria_log_delay)
s2 <- sd(maria_log_delay)
n2 <- length(maria_log_delay)

ggplot(maria_result, aes(x = log_delay)) +
  geom_histogram(aes(y = after_stat(density)), bins = n_bins, fill = "grey", col = "black") + 
  geom_density(col = "red", lwd = 1) + 
  stat_function(fun = dnorm, args = list(mean = xbar2, sd = s2), col="blue", lwd = 1) + 
  ggtitle("Histogram of Ln transformed Maria Delay") +
  xlim(0, 17) +
  ylim(0, 0.6)

ggplot(psql_result, aes(x = log_delay)) +
  geom_histogram(aes(y = after_stat(density)), bins = n_bins, fill = "grey", col = "black") + 
  geom_density(col = "red", lwd = 1) + 
  stat_function(fun = dnorm, args = list(mean = xbar1, sd = s1), col="blue", lwd = 1) + 
  ggtitle("Histogram of Ln transformed Postgres Delay") +
  xlim(0, 17) +
  ylim(0, 0.6)

# More hypothesis testing with log transformed data, no difference in results
tts <- (xbar1 - xbar2) / sqrt(s1 * s1 / n1 + s2 * s2 / n2)
df <- ((s1 * s1 / n1 + s2 * s2 / n2) ^ 2) /
  ((s1 * s1 / n1) ^ 2 / (n1 - 1) +
     (s2 * s2 / n2) ^ 2 / (n2 - 1))
p_value <- 2 * pt(-abs(tts), df)
# 0

# Overlapping histogram
data <- data.frame(
  value = c(psql_log_delay, maria_log_delay),
  group = factor(rep(c("PSQL Log Delay", "Maria Log Delay"), 
                     times = c(length(psql_log_delay), length(maria_log_delay))))
)
ggplot(data, aes(x = value, fill = group)) +
  geom_histogram(aes(y = ..density..), position = "identity", alpha = 0.5, binwidth = 0.5) +
  labs(title = "Overlapping Histograms  of Database Histograms",
       x = "log_delay",
       y = "Density") +
  scale_fill_manual(values = c("blue", "red")) +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "#F5F5F5")) +
  xlim(0, 17) + 
  ylim(0, 0.6)

# Disconnect from database
dbDisconnect(psql_con)
dbDisconnect(maria_con)
