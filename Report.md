# Report
### Learning Algorithm
I implemented MADDPG. I chose MADDPG because its usefulness in different types of multi agent environments, involving cooperation, competition and mixed environments. It is truly a general purpose multi agent algorithm. I started with my last project as base using ddpg (https://github.com/juanluislopez24/robot-arm-reacher). I modified it according to the MADDPG paper. The main difference is in the critic network, since it receives information from other agents on the environment.

The hyperparameters I chose are:
- BUFFER_SIZE = int(1e6)  
- BATCH_SIZE = 256   
- GAMMA = 0.99       
- TAU = 2e-3      
- LR_ACTOR = 1.0e-3     
- LR_CRITIC = 1.0e-3     
- WEIGHT_DECAY = 0
- UPDATE_EVERY = 10
- UPDATE_NUM = 2
- EPSILON = 1.0
- EPS_DECAY = 1.0e-5
- EPS_END=0.02


The critic architecture is as follows:
- input = state_size * number_of_agents (24*2)
- First fully connected layer (input 48, output 256)
- Second fully connected layer (input 256 + action_size*number_of_agents (4 * 2), output 128)
- Third fully connected layer (input 128, output 1)

The actor architecture is as follows:
- input = state_size (24)
- First fully connected layer (input 24, output 256)
- Second fully connected layer (input 256, output 128)
- Third fully connected layer (input 128, output action_size (2))

I trained the agent for 3037 episodes, updating the  weights 2 times every 10 steps. One of the agents reached an average score during the last 100 episodes of 0.51. The agent turned very consistent when testing, reaching scores of at least +2.2 on each episode. 


### Plot of Rewards
[image1]: scores1.PNG "rewards"
[image2]: scores2.PNG "rewards"
Scores for the first agent

![Rewards][image1]

Scores for the second agent

![Rewards][image2]

As you can see both agents where consistently reaching scores of 2.0 after training episode ~2800.


### Ideas for Future Work
This is a basic implementation for MADDPG. Prioritized experience replay could be added so the agents can learn their most significant mistakes. As the paper states, for each agent we could have an ensemble of policies (actors), which we take an average of them all. Also if the number of agents increases, our critic network's parameters increases, we could consider only taking into account information from close or strategic positioned agents. 