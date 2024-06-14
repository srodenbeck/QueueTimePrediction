### Constants
- Epochs: 40
- Optimizer: Adam
- Loss: CrossEntropyLoss
- 10% test then 80% train 20% val
### Hyperparameters
- Feature_transform: none, log, min-max
- Features: memory, cpu, job_count, queue, running, all, queue_request
- Learning Rate: 1e-5 to 1e-1
- Model Architecture
	- Num Layers: 1 to 3
	- Layer Size: 32 to 180
	- Activation Function: relu, leaky_relu, tanh, elu
	- Dropout: 0.1 to 0.5

### 2024-06-13 Notes on Recent Jobs
- Relu and leaky relu are best
- Transform min max and sometimes log
- 2 layers with 1 or 2 of them being very large
- 0.001ish learning rate ver consistently
- all features every time
- Dropout values between 0.12 and 0.4
- Model is likely overfitting
##### Changes to be made
- Make learning rate fixed at 0.001
- Make it transform min max everytime
- Use relu and leaky relu only
- Condense dropout options
- Remove first 3 feature collections

