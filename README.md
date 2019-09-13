# DQN_simple_games
Solving simple openAi gym games using reinforcement learning and neural networks.

## Games

### Mountain Car

The goal of this simple game is to climb a 'mountain'.
Action space is really basic since the agent choose between 3 actions:
- Go left
- Go Right
- Do Nothing

The inputs are the position and the velocity of the car.

The agent, with no prior knowledge of the environment will understand that in
order to climb the mountain, it has to build momentum by alternatively going 
right and left.
