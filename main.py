import torch
import time
import matplotlib.pyplot as plt
from Frameworks import DDPG
import gym
from collections import deque
from Frameworks import PPO_clip_agent


def main():
	# arguments
	pnetwork_mid_dims = [256, 128, 64]
	qnetwork_mid_dims = [256, 128, 64]
	test_epochs = min(int(200 * 0.01), 20)

	# Environment
	env_name = "Pendulum-v0"  # "MountainCarContinuous-v0-Pendulum"
	env = gym.make(env_name)
	action_space = env.action_space.shape[0]
	action_space_high = env.action_space.high
	action_space_low = env.action_space.low
	observation_space = env.observation_space.shape[0]

	# Agent
	agent = PPO_clip_agent(observation_space,action_space, pnetwork_mid_dims, v)
	# agent = DDPG(pnetwork_mid_dims, qnetwork_mid_dims, action_space, action_space_high, action_space_low,
	# 			 observation_space, lr=0.01)
	#
	# agent.freeze_target_networks()
	# agent.run(env)
	env.close()

	# total_steps = 0
	# rewards_list = []
	# score_deque = deque(maxlen=100)
	# for i in range(epochs):
	# 	observation = env.reset()
	# 	done = False
	# 	j = 0
	# 	game_reward = []
	# 	while (not done) and j < max_steps_per_episode:
	# 		env.render()
	# 		if total_steps > random_actions_till:
	# 			action = agent.take_action(observation)
	# 		else:
	# 			action = torch.FloatTensor(env.action_space.sample())
	# 		next_observation, reward, done, _ = env.step(action)
	# 		game_reward.append(reward)
	# 		score_deque.append(reward)
	# 		agent.ReplayBuffer(observation, action, reward, next_observation, done)
	# 		observation = next_observation
	#
	# 		j += 1
	# 		total_steps += 1
	#
	# 	if total_steps > update_after and total_steps % update_every == 0:
	# 		for k in range(no_of_updates):
	# 			batch = agent.ReplayBuffer.sample(batch_size)
	# 			agent.UpdateQ(batch)
	# 			agent.UpdateP(batch)
	# 			agent.UpdateNetworks()
	#
	# 	avg_reward_this_game = sum(game_reward) / len(game_reward)
	# 	with open("episode_reward_ddpg.txt", "a+") as f:
	# 		f.write(str(sum(game_reward)) + "\n")
	# 	rewards_list.append(avg_reward_this_game)
	# 	print(
	# 		f'For game number {i},avg reward this game {avg_reward_this_game}, mean of last 100 rewards = {sum(score_deque) / 100}')
	# 	# env.close()
	#
	# # Plotting avg rewards per game
	# plt.figure(figsize=(8, 6))
	# plt.title("Average reward of DDPG agent on" + env_name + "for each game")
	# plt.plot(range(len(rewards_list)), rewards_list)
	# plt.savefig("figures/DDPG_" + env_name + "_rewards.png")
	# plt.show()
	#
	# for i_ in range(test_epochs):
	# 	with torch.no_grad():
	# 		observation = env.reset()
	# 		done = False
	# 		j_ = 0
	# 		while not (done or j_ > test_steps):
	# 			env.render()
	# 			time.sleep(1e-3)
	# 			action = agent.take_action(observation)
	# 			observation, _, done, _ = env.step(action)
	# 			j_ += 1
	# 		env.close()


if __name__ == "__main__":
	torch.manual_seed(171)
	main()