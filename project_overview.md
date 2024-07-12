## Directories
- docs/
  - This directory contains documentation and notes for the project
- graphs/
  - This directory contains an assorted collection of graphs
  - boxplots_by_time/: Feature box plots grouped by queue time
  - Graphs/: Queue time averages by month on ANVIL overtime
  - TrendLines/: Trend lines correlating features and queue time
- old_vs_new_graphs/ 
  - This directory contains histograms comparing the Maria DB job database to data from ANVIL
- scrap/ 
  - This directory contain nonessential scripts and left over files 
- scripts/ 
  - This directory contains programs used for graphing and data analysis
- utils/
  - This directory contains a few utility files for data processing

## Files
- aus_regression.py: PyTorch neural network regression model training incorporating optuna hyperparameter optimization
- calculate_features.py: Contains methods for calculating additional features based on original sacct job info
- classify_train.py: Training pytorch neural network binary classification model on jobs
- database.py: Program used for uploading csv files obtained from Slurm sacct to postgres database
- model.py: Contains methods for various neural network model architectures used
- optuna_train.py: Potential program for finding optimal regression neural network hyperparameters with Optuna
- read_db.py: Contains methods for connecting to and reading from PostgreSQL database
- train.py: File for training PyTorch neural network regression models
- transformations: Contains methods for transforming and normalizing job data
- trout.py: Command line file to be deployed for use by ANVIL users
- xgboost_model.py: Program for creating basic XGBoost regression model to test with