# DeepPPO AtariAsteroids NoFrameSkipV4
Deep Proximal Policy Optimiser RL architecture for Atari Asteroids NoFrameSkipV4 provided by Gymnasium. Implemented in PyTorch.
## Architecture
![Architecture diagram.](Illustrations/architecture_diagram.png)
Backbone of the architecture consists of NatureCNN from the paper "Human-level control through deep reinforcement learning" by Volodymyr Mnih, Koray Kavukcuoglu, David Silver et al. It was chosen for its lightweight nature, and the fact, that it was developed exactly for such tasks. 
