[KARNAC](https://purdue0-my.sharepoint.com/:b:/g/personal/srodenb_purdue_edu/EbpbLzEhyo9Hr4p1FSdXq6IBdeubL2E2xpf9B5QvaRKNsw?e=cAyycf)
## Mastering HPC Runtime
- [link](https://www.nrel.gov/docs/fy23osti/86526.pdf) [repo](https://github.com/NREL/eagle-jobs) [OEDI](https://data.openei.org/submissions/5860)
- Number of jobs ranges from 7K to 19M
	- Majority between 100K and 2M
- XGboost XGBRegressor model with default hyperparameters
- TensorFlow NN with 3 fully connected layers of 64 nodes and ReLU activations.
	- Adam optimizer, lr=0.001, batch_size=10,000 jobs, epochs=10
	- Early stopping with a patience of 1 epoch (?)
- TF-IDF model using sklearn
##### Reccomendations
- Split based on submit time. Training data should be from a time interval prior to test data.
- Investigate the data, requested wallclock are inaccurate.
- Consider the encoding method for categorical data. Authors found that label encoding was better than one-hot encoding.
	- For one-hot encoding, only consider the top $n$ most common instances for each feature in order to reduce dimensionality
- Try different time windows for training and test data
- Time series heuristics (i.e. looking at recent and similar jobs runtime) are less effective than normal machine learning methods
- Context is important for error metrics. Absolute error metrics are better than plain accuracy.
- Finding the optimal feature set for the previous split time and using it to determine the feature set is intuitive but was marginally worse than the normal overall method.
## Imbalanced Regression
- [Neptune](https://neptune.ai/blog/how-to-deal-with-imbalanced-classification-and-regression-data)
- [Medium for paper](https://towardsdatascience.com/strategies-and-tactics-for-regression-on-imbalanced-data-61eeb0921fca)
- [smogn](https://pypi.org/project/smogn/)