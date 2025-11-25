"""
PPO (Proximal Policy Optimization) Reinforcement Learning Model
Optimizes race strategy using the IL model as initialization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import json
from pathlib import Path
import pickle

class RacingEnvironment:
    """Simulated racing environment for RL training"""
    def __init__(self, il_model, lstm_model, track_length=144672):
        self.il_model = il_model  # Imitation Learning model for initialization
        self.lstm_model = lstm_model  # LSTM for tire degradation prediction
        self.track_length = track_length
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.lap = 1
        self.lap_distance = 0.0
        self.position = 1  # Starting position
        self.tire_age = 0  # Laps on current tires
        self.total_time = 0.0
        self.pit_stops = 0
        self.done = False
        
        # Initial state
        state = self.get_state()
        return state
    
    def get_state(self):
        """Get current state vector"""
        # State: [lap, lap_progress, tire_age, position, predicted_degradation, ...]
        lap_progress = self.lap_distance / self.track_length
        
        # Predict tire degradation using LSTM (simplified)
        degradation = 0.0  # Will be computed from LSTM if available
        
        state = np.array([
            self.lap / 30.0,  # Normalized lap number
            lap_progress,  # Progress in current lap
            self.tire_age / 20.0,  # Normalized tire age
            self.position / 25.0,  # Normalized position
            degradation,  # Predicted degradation
            self.pit_stops / 3.0  # Normalized pit stops
        ])
        
        return state
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        # Action: [pit_decision (0=stay out, 1=pit), driving_aggression (0-1)]
        pit_decision = action[0] > 0.5
        driving_aggression = np.clip(action[1], 0.0, 1.0)
        
        reward = 0.0
        
        # Handle pit stop
        if pit_decision and self.lap_distance < 0.1:  # Only pit at start/finish
            self.pit_stops += 1
            self.tire_age = 0
            self.total_time += 34.0  # Pit stop time (34 seconds from track info)
            reward -= 5.0  # Penalty for pit stop
        
        # Advance lap
        if self.lap_distance >= self.track_length:
            self.lap += 1
            self.lap_distance = 0.0
            self.tire_age += 1
            
            # Calculate lap time (simplified - based on tire age and aggression)
            base_lap_time = 97.0  # Base lap time in seconds (~1:37)
            tire_penalty = self.tire_age * 0.5  # 0.5s per lap degradation
            aggression_bonus = (1.0 - driving_aggression) * 2.0  # Less aggression = faster
            
            lap_time = base_lap_time + tire_penalty - aggression_bonus
            self.total_time += lap_time
            
            # Position update (simplified)
            if driving_aggression > 0.7:
                # Aggressive driving might gain positions but risk errors
                if np.random.random() > 0.1:  # 90% chance of gaining position
                    self.position = max(1, self.position - 1)
                    reward += 2.0
                else:
                    # 10% chance of error
                    self.position = min(25, self.position + 1)
                    reward -= 5.0
            else:
                # Conservative driving maintains position
                reward += 0.5
        
        # Update lap distance
        speed = 50.0 + driving_aggression * 30.0  # Simplified speed
        self.lap_distance += speed * 0.1  # Advance by speed * dt
        
        # Check if race is done
        if self.lap > 27:  # Race length
            self.done = True
            # Final reward based on position
            position_reward = (26 - self.position) * 10.0
            time_penalty = -self.total_time / 10.0
            reward += position_reward + time_penalty
        
        # Intermediate rewards
        reward += 0.1  # Small reward for continuing
        
        next_state = self.get_state()
        return next_state, reward, self.done, {}

class PolicyNetwork(nn.Module):
    """Policy network for PPO"""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Separate heads for mean and std
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.std_head = nn.Linear(input_dim, action_dim)
        
        # Initialize with small values
        nn.init.uniform_(self.std_head.weight, -1e-3, 1e-3)
        nn.init.constant_(self.std_head.bias, -1.0)
    
    def forward(self, state):
        shared = self.shared(state)
        mean = torch.tanh(self.mean_head(shared))  # Actions in [-1, 1]
        std = torch.clamp(torch.exp(self.std_head(shared)), 0.1, 1.0)
        return mean, std

class ValueNetwork(nn.Module):
    """Value network for PPO"""
    def __init__(self, state_dim, hidden_dims=[256, 256]):
        super(ValueNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)

class PPOAgent:
    """PPO Agent for race strategy optimization"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, value_coef=0.5, entropy_coef=0.01, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        self.memory = deque(maxlen=10000)
    
    def initialize_from_il(self, il_model_path=None, state_dim=None, action_dim=None):
        """Initialize policy network from Imitation Learning model"""
        print("Initializing PPO policy from IL model...")
        # Copy weights from IL model to policy network (if compatible)
        # This is a simplified version - in practice, you'd need to map layers appropriately
        try:
            if il_model_path and Path(il_model_path).exists():
                # Load IL model checkpoint
                il_checkpoint = torch.load(il_model_path, map_location=self.device)
                policy_state_dict = self.policy.state_dict()
                
                # Copy compatible layers (shared layers)
                for name, param in il_checkpoint.items():
                    if name in policy_state_dict and param.shape == policy_state_dict[name].shape:
                        policy_state_dict[name] = param
                
                self.policy.load_state_dict(policy_state_dict, strict=False)
                print("Successfully initialized from IL model")
            else:
                print("IL model path not found, starting with random initialization")
        except Exception as e:
            print(f"Could not initialize from IL model: {e}")
            print("Starting with random initialization")
    
    def select_action(self, state, deterministic=False):
        """Select action using policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():  # Disable gradient computation for inference
            mean, std = self.policy(state_tensor)
            
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=1)
        
        action = action.cpu().detach().numpy()[0]  # Use detach() to remove from computation graph
        action = np.clip(action, -1.0, 1.0)
        
        if deterministic:
            return action, 0.0
        else:
            return action, log_prob.item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store transition in replay buffer"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })
    
    def compute_returns(self, rewards, dones, next_value=0.0):
        """Compute discounted returns"""
        returns = []
        G = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0.0
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns
    
    def update(self, batch_size=64, epochs=10):
        """Update policy and value networks"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = torch.FloatTensor([b['state'] for b in batch]).to(self.device)
        actions = torch.FloatTensor([b['action'] for b in batch]).to(self.device)
        old_log_probs = torch.FloatTensor([b['log_prob'] for b in batch]).to(self.device)
        rewards = [b['reward'] for b in batch]
        dones = [b['done'] for b in batch]
        next_states = torch.FloatTensor([b['next_state'] for b in batch]).to(self.device)
        
        # Compute returns
        next_values = self.value(next_states).detach().cpu().numpy().flatten()
        returns = self.compute_returns(rewards, dones, next_values[-1] if len(next_values) > 0 else 0.0)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Compute advantages
        values = self.value(states).squeeze()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy
        for _ in range(epochs):
            mean, std = self.policy(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1).mean()
            
            # PPO clip
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            policy_loss -= self.entropy_coef * entropy
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
        
        # Update value
        for _ in range(epochs):
            values = self.value(states).squeeze()
            value_loss = nn.MSELoss()(values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.value_optimizer.step()
    
    def save(self, model_path):
        """Save agent"""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, model_path)
        print(f"PPO agent saved to {model_path}")
    
    def load(self, model_path):
        """Load agent"""
        checkpoint = torch.load(model_path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        print(f"PPO agent loaded from {model_path}")

def train_ppo_agent(il_model_path=None, episodes=1000, output_dir='models'):
    """Train PPO agent"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create environment
    env = RacingEnvironment(il_model=None, lstm_model=None)
    state_dim = len(env.get_state())
    action_dim = 2  # [pit_decision, driving_aggression]
    
    # Create agent
    agent = PPOAgent(state_dim, action_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize from IL if available
    if il_model_path:
        agent.initialize_from_il(il_model_path, state_dim, action_dim)
    
    # Training loop
    episode_rewards = []
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < 1000:
            action, log_prob = agent.select_action(state)
            value = agent.value(torch.FloatTensor(state).unsqueeze(0).to(agent.device)).item()
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            agent.store_transition(state, action, reward, next_state, done, log_prob, value)
            
            state = next_state
            steps += 1
        
        # Update agent
        if len(agent.memory) >= 64:
            agent.update(batch_size=64, epochs=5)
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}")
    
    # Save agent
    agent.save(output_path / 'ppo_agent.pth')
    
    return agent, episode_rewards

if __name__ == "__main__":
    train_ppo_agent(episodes=1000)

