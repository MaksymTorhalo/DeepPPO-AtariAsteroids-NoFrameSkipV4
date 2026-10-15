num_envs = 8
policy_gain = 0.01
value_gain = 1.0
num_steps = 256
gamma = 0.99
gae_lambda = 0.95
clip_coef = 0.1
ent_coef = 0.01
val_coef = 0.5
max_grad_norm = 0.5
lr = 2.5e-4
adam_eps = 1e-5
timesteps = 20000000
batch_size = num_envs * num_steps
num_minibatches = 4
minibatch_size = batch_size // num_minibatches
num_updates = timesteps // batch_size
update_epochs = 4




SEED = 42