import torch
import torch.nn as nn
from policy_network import PolicyNetwork

class ReinforceAgent:
    def __init__(self, gamma=0.95):
        self.gamma = gamma  
        self.log_probs = []
        self.rewards = []
    
    def reset(self):
        # Clear data for new episode
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, state, policy_net):
        action_probs = policy_net(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample() # gets a random action based on probabilities
        self.log_probs.append(dist.log_prob(action)) #append the log prob of that action
        return action.item()
    
    def calculate_discounted_rewards(self):
        discounted_rewards = []
        G = 0
        


        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            discounted_rewards.insert(0, G)
        
        return torch.tensor(discounted_rewards, dtype=torch.float32)
    
    def calculate_policy_loss(self):
        if len(self.rewards) == 0:
            return torch.tensor(0.0, requires_grad=True)
            
        returns = self.calculate_discounted_rewards()
        
        # Normalize rewards (reduces variance)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate policy gradient terms
        policy_gradient_terms = torch.stack(self.log_probs) * returns
        
        # here we are maximizing reward thats why we had negative sign because in neural networks we minimize loss
        loss = -policy_gradient_terms.sum()
        
        return loss
    
    def collect_episode(self, env, policy_net, max_steps=100):
        #Running one episode and collecting data
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = self.select_action(state, policy_net)
            state, reward, done, info = env.step(action)
            self.rewards.append(reward)
            total_reward += reward
            
            if done:
                break
        
        return total_reward, step + 1
