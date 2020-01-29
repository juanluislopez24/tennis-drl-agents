[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Tennis DRL: competition and collaboration

Deep reinforcement agents which have to collaborate in order to play a game for a period of time, and competing to get the best score. This is part of the third and final graded project for the Deep Reinforcement Learning Nanodegree on Udacity.

In this environment there is a tennis court with 2 agents, one on each side. The agents control rackets to bounce a ball over a net. If an agent passes the ball over the net, it receives a reward of +0.1. If an agent lets the ball hit the ground on their side, it receives a reward of -0.01. So the agent's goal is to keep the ball from falling and keep playing.

For each agent, the environment consists of a state space of 24 different variables, corresponding to a local of observation from an agent. The action space consists of 2 different continuous actions, corresponding to horizontal and vertical movements.

The environment is considered solved, when the average over 100 episodes of an agent is at least +0.5.

I implemented a multi agent deep deterministic policy gradient approach (MADDPG) since it comes natural to use in a general purpose multi agent environment.


![Trained Agent][image1]

### Steps to get this thing going
1. Create a new environment with python version 3.6.x `conda create -n tennis python=3.6`
2. Activate your environment `conda activate tennis`
3. Install mlagents `pip install mlagents==0.4.0`
4. Install pytorch, get your command from https://pytorch.org/get-started/locally/ in my case it is `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
5. Install jupyter lab (it looks good) `pip install jupyterlab`
6. Clone this repository `git clone https://github.com/juanluislopez24/robot-arm-reacher`
7. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

8. Uncompress the zip file on your desired folder

9. Open jupyter lab with `jupyter lab` and open the Tennis.ipynb file
10. Change the path for your unity environment on the cell that has `env = UnityEnvironment(file_name='env_v1/Tennis.exe')`
11. Run all the cells and train a new agent, or skip training and go down until the section of "Run the trained agent" and run all the remaining cells.


Have fun and thanks for passing through here!
