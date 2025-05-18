import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import json
import logging
from collections import deque, namedtuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Save directory for models
MODEL_SAVE_DIR = os.path.join('app', 'static', 'models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Experience tuple for DQN replay buffer
Experience = namedtuple('Experience',
                        field_names=['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity=10000):
        """Initialize replay buffer with given capacity."""
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from buffer."""
        # Ensure batch_size is not larger than current buffer size
        actual_batch_size = min(batch_size, len(self.buffer))
        if actual_batch_size <= 0:
            return None # Cannot sample from empty buffer

        indices = np.random.choice(len(self.buffer), actual_batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for idx in indices:
            experience = self.buffer[idx]
            states.append(experience.state)
            actions.append(experience.action)
            rewards.append(experience.reward)
            next_states.append(experience.next_state)
            dones.append(experience.done)

        # Convert to tensors
        # Use torch.as_tensor for potentially better performance, ensure float32
        # Handle potential non-numpy states robustly
        states = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in states]).to(device)
        actions = torch.as_tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        # Handle potential non-numpy next_states
        next_states = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in next_states]).to(device)
        dones = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)


        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current size of replay buffer."""
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network with layer normalization and dropout."""

    def __init__(self, input_dim, output_dim, hidden_dim=256):
        """Initialize DQN with given input and output dimensions."""
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        """Forward pass through the network."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(device)
        # Ensure input tensor is on the correct device
        x = x.to(next(self.parameters()).device)
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


class DQNAgent:
    """DQN agent with replay buffer and target network."""

    def __init__(self, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, epsilon_start=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64,
                 target_update_freq=10, name="dqn"):
        """Initialize DQN agent with parameters."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size # Configurable batch size
        self.target_update_freq = target_update_freq
        self.name = name

        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Learning steps counter
        self.learn_step_counter = 0

        # Average Q-values for elbow method
        self.q_values_history = []
        self.avg_q_values = None

        logger.info(f"Initialized {name} agent (State dim: {state_dim}, Action dim: {action_dim}, Batch Size: {self.batch_size})")

    def select_action(self, state, explore=True):
        """Select action using epsilon-greedy policy."""
        if explore and np.random.random() < self.epsilon:
            # Exploration
            return np.random.randint(self.action_dim)
        else:
            # Exploitation
            self.policy_net.eval()
            with torch.no_grad():
                # Ensure state is a tensor on the correct device
                if not isinstance(state, torch.Tensor):
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                else:
                    state_tensor = state.unsqueeze(0).to(device) if state.ndim == 1 else state.to(device)

                q_values = self.policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()
            self.policy_net.train()
            return action

    def select_action_elbow(self, state, top_k=None):
        """Select actions using elbow method on Q-values."""
        self.policy_net.eval()
        with torch.no_grad():
            # Ensure state is a tensor on the correct device
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                 state_tensor = state.unsqueeze(0).to(device) if state.ndim == 1 else state.to(device)

            # Get Q-values
            q_values = self.policy_net(state_tensor).squeeze().cpu().numpy()

            # Handle case where q_values might be scalar if action_dim is 1
            if not isinstance(q_values, np.ndarray) or q_values.ndim == 0:
                 if isinstance(q_values, (float, int, np.number)): # Check if it's a number
                      q_values = np.array([q_values]) # Convert scalar to array
                 else:
                      logger.warning(f"Invalid Q-values type/shape for elbow method: {type(q_values)}, shape={getattr(q_values, 'shape', 'N/A')}")
                      self.policy_net.train()
                      return [0] if self.action_dim > 0 else [] # Fallback: return first action or empty

            # Save Q-values for averaging
            self.q_values_history.append(q_values)
            if len(self.q_values_history) > 100:  # Keep history limited
                self.q_values_history.pop(0)

            # Calculate average Q-values
            try:
                self.avg_q_values = np.mean(self.q_values_history, axis=0)
            except Exception as e:
                 logger.error(f"Error calculating average Q-values: {e}", exc_info=True)
                 self.avg_q_values = q_values # Fallback to current Q-values

            # Ensure avg_q_values has the correct dimension
            if self.avg_q_values.shape != (self.action_dim,):
                 if self.avg_q_values.size == self.action_dim:
                      self.avg_q_values = self.avg_q_values.reshape(self.action_dim)
                 else:
                      logger.warning(f"Avg Q-values shape mismatch ({self.avg_q_values.shape}) vs action_dim ({self.action_dim}). Using current Q-values.")
                      self.avg_q_values = q_values # Fallback


            # Find elbow point or use top_k if specified
            if top_k is not None:
                # Just return top k actions based on avg Q-values
                # Ensure indices are int
                top_indices = np.argsort(self.avg_q_values)[::-1][:top_k]
                return [int(i) for i in top_indices]
            else:
                # Use elbow method to find natural cutoff
                sorted_indices = np.argsort(self.avg_q_values)[::-1]
                sorted_values = self.avg_q_values[sorted_indices]

                if len(sorted_values) <= 1:
                    return [int(sorted_indices[0])] if len(sorted_indices) > 0 else []

                # Calculate differences between adjacent Q-values
                diffs = np.diff(sorted_values)

                # Avoid issues with zero differences if all values are identical
                if np.allclose(diffs, 0):
                     # If all values are same, return top 1 or maybe top few? Let's return top 1.
                     return [int(sorted_indices[0])]

                # Find the elbow point (maximum curvature - approximated by largest diff change)
                # Add a small epsilon to avoid division by zero if diffs contains zero
                diffs_nonzero = diffs + 1e-9
                # Calculate second differences (change in slope)
                diffs2 = np.diff(diffs_nonzero)

                # Elbow is where the second difference is minimized (most negative - sharpest turn)
                # Add 1 because np.diff reduces length, +1 to get index *after* the turn
                elbow_idx = np.argmin(diffs2) + 1 + 1
                elbow_idx = max(1, min(elbow_idx, len(sorted_indices))) # Clamp index

                # Return indices up to the elbow point, ensuring ints
                return [int(i) for i in sorted_indices[:elbow_idx]]

        self.policy_net.train() # Ensure model is back in train mode

    def store_experience(self, state, action, reward, next_state, done=False):
        """Store experience in replay buffer."""
        # Ensure data types are suitable (e.g., numpy arrays or lists)
        # State/Next State should ideally be numpy arrays for consistency
        if isinstance(state, torch.Tensor): state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor): next_state = next_state.cpu().numpy()
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update_epsilon(self):
        """Decay epsilon value for exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self):
        """Update network parameters using experiences from replay buffer."""
        # Check if enough samples in buffer FOR THE CONFIGURED BATCH SIZE
        if len(self.replay_buffer) < self.batch_size:
            # logger.debug(f"Skipping DQN learn: Buffer size ({len(self.replay_buffer)}) < Batch size ({self.batch_size})")
            return 0.0 # Return 0 loss if not enough samples

        # Sample batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
             logger.warning("Replay buffer sample returned None. Skipping learn step.")
             return 0.0

        states, actions, rewards, next_states, dones = batch

        # --- Q-Value Calculation ---
        # Get Q(s, a) for the actions taken
        # policy_net output shape: [batch_size, action_dim]
        # actions shape: [batch_size, 1]
        # gather(1, actions) selects the Q-value corresponding to the action taken in each state
        q_values = self.policy_net(states).gather(1, actions)

        # --- Target Q-Value Calculation (Double DQN) ---
        with torch.no_grad():
            # 1. Select best action a' for Q(s', a') using the *policy* network
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # 2. Evaluate Q(s', a') for that action a' using the *target* network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            # 3. Calculate target: r + gamma * Q_target(s', a') * (1 - done)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # --- Loss Calculation ---
        # Use Smooth L1 Loss (Huber Loss) - less sensitive to outliers than MSE
        loss = F.smooth_l1_loss(q_values, target_q_values)

        # --- Optimization ---
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # --- Target Network Update ---
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # logger.debug(f"Updated DQN target network at step {self.learn_step_counter}")

        return loss.item() # Return scalar loss value

    def save_model(self):
        """Save model to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_SAVE_DIR, f"{self.name}_{timestamp}.pt")

        try:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'q_values_history': self.q_values_history, # Save history for potential analysis
                'learn_step_counter': self.learn_step_counter,
                'batch_size': self.batch_size # Save config used
            }, model_path)
            logger.info(f"Saved {self.name} model to {model_path}")
            return model_path
        except Exception as e:
             logger.error(f"Error saving {self.name} model to {model_path}: {e}", exc_info=True)
             return None

    def load_model(self, model_path):
        """Load model from disk."""
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found")
            return False

        try:
            # Load checkpoint onto the correct device
            checkpoint = torch.load(model_path, map_location=device)

            # Basic dimension check before loading state dicts
            if checkpoint.get('state_dim') != self.state_dim or checkpoint.get('action_dim') != self.action_dim:
                 logger.error(f"Dimension mismatch loading model {model_path}. Expected ({self.state_dim}, {self.action_dim}), got ({checkpoint.get('state_dim')}, {checkpoint.get('action_dim')}). Load aborted.")
                 return False

            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min) # Use saved epsilon or min
            self.learn_step_counter = checkpoint.get('learn_step_counter', 0)
            self.q_values_history = checkpoint.get('q_values_history', [])
            # Optionally load and log batch size if needed
            loaded_batch_size = checkpoint.get('batch_size', self.batch_size)
            if loaded_batch_size != self.batch_size:
                logger.warning(f"Loaded model batch size ({loaded_batch_size}) differs from current config ({self.batch_size}).")
                # Decide whether to update self.batch_size or just log

            self.policy_net.to(device) # Ensure models are on the correct device
            self.target_net.to(device)
            self.target_net.eval() # Ensure target net is in eval mode


            logger.info(f"Loaded {self.name} model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}", exc_info=True)
            return False


# Helper for layer initialization
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for linear layers."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """PPO Actor network for continuous LLM parameters (temperature, max_length)."""

    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.shared_layers = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh()
        )
        self.temp_mean = layer_init(nn.Linear(64, 1), std=0.01)
        self.temp_log_std = nn.Parameter(torch.zeros(1)) # Learnable log std deviation
        self.len_mean = layer_init(nn.Linear(64, 1), std=0.01)
        self.len_log_std = nn.Parameter(torch.zeros(1)) # Learnable log std deviation

    def forward(self, state):
        """Forward pass returning means and stdevs of actions."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(device)
        # Ensure input tensor is on the correct device
        state = state.to(next(self.parameters()).device)

        x = self.shared_layers(state)

        # Temperature mean (output of linear layer, then squashed by tanh)
        temp_mean = torch.tanh(self.temp_mean(x)) # Mean in range [-1, 1]
        # Temperature std dev (exponentiated learnable parameter)
        temp_log_std = self.temp_log_std.expand_as(temp_mean) # Ensure shape matches mean
        temp_std = torch.exp(temp_log_std)

        # Length mean (output of linear layer, then squashed by tanh)
        len_mean = torch.tanh(self.len_mean(x)) # Mean in range [-1, 1]
        # Length std dev (exponentiated learnable parameter)
        len_log_std = self.len_log_std.expand_as(len_mean) # Ensure shape matches mean
        len_std = torch.exp(len_log_std)

        # Add small epsilon for numerical stability (prevent std=0)
        epsilon = 1e-6

        return temp_mean, temp_std + epsilon, len_mean, len_std + epsilon


class Critic(nn.Module):
    """PPO Critic network to estimate state value V(s)."""

    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0) # Critic output layer often initialized differently
        )

    def forward(self, state):
        """Forward pass returning state value."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(device)
         # Ensure input tensor is on the correct device
        state = state.to(next(self.parameters()).device)

        return self.critic(state)


class PPOAgent:
    """PPO Agent managing Actor, Critic, Memory, and Learning updates."""

    def __init__(self, state_dim, temp_range=(0.1, 1.0), len_range=(50, 2000),
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=10,
                 eps_clip=0.2, gae_lambda=0.95, entropy_coef=0.01, name="ppo"):
        """Initialize the PPO agent with parameters."""
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.name = name

        # Action scaling ranges
        self.temp_min, self.temp_max = temp_range
        self.len_min, self.len_max = len_range

        # Initialize networks
        self.policy = Actor(state_dim).to(self.device)
        self.policy_old = Actor(state_dim).to(self.device) # For calculating ratio
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.critic = Critic(state_dim).to(self.device)

        # Initialize optimizers
        self.optimizer_actor = optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Initialize memory buffer (stores experiences for one update cycle)
        self.memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': [], 'values': []}
        self.memory_size = 0

        logger.info(f"Initialized {name} agent (State dim: {state_dim})")

    def _scale_action(self, tanh_action, min_val, max_val):
        """Scale tanh action (-1 to 1) to the specified range."""
        # (tanh_action + 1) / 2 maps [-1, 1] to [0, 1]
        # Then scale to [min_val, max_val]
        return min_val + (tanh_action + 1.0) * 0.5 * (max_val - min_val)

    def _unscale_action(self, scaled_action, min_val, max_val):
        """Convert action from specified range back to tanh range (-1 to 1)."""
        if not isinstance(scaled_action, torch.Tensor):
            # Convert numpy array or scalar to tensor
            scaled_action = torch.as_tensor(scaled_action, dtype=torch.float32, device=self.device)
        # Map [min_val, max_val] to [0, 1]
        normalized_action = (scaled_action - min_val) / (max_val - min_val)
        # Map [0, 1] to [-1, 1]
        return 2.0 * normalized_action - 1.0

    def select_action(self, state):
        """Select temperature and max_length parameters based on state."""
        with torch.no_grad():
            # Ensure state is tensor and on correct device
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            else:
                state_tensor = state.to(self.device)

            # Add batch dimension if missing
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)

            # Get action distributions from the 'old' policy (for consistent sampling during collection)
            temp_mean_tanh, temp_std, len_mean_tanh, len_std = self.policy_old(state_tensor)

            # Create Normal distributions
            temp_dist = Normal(temp_mean_tanh, temp_std)
            len_dist = Normal(len_mean_tanh, len_std)

            # Sample actions in the tanh space [-1, 1]
            temp_action_tanh = temp_dist.sample()
            len_action_tanh = len_dist.sample()

            # Calculate log probabilities of the sampled actions
            temp_log_prob = temp_dist.log_prob(temp_action_tanh)
            len_log_prob = len_dist.log_prob(len_action_tanh)
            # Sum log probs for joint probability (assuming independence)
            log_prob = temp_log_prob + len_log_prob

            # Get state value estimate from the critic
            value = self.critic(state_tensor)

        # Store experience components in memory (as CPU tensors/scalars for easier handling later)
        # Squeeze batch dimension before storing
        self.memory['states'].append(state_tensor.squeeze(0).cpu())
        # Store action in tanh space
        self.memory['actions'].append(torch.cat([temp_action_tanh, len_action_tanh], dim=-1).squeeze(0).cpu())
        self.memory['logprobs'].append(log_prob.squeeze().cpu())
        self.memory['values'].append(value.squeeze().cpu())
        self.memory_size += 1

        # Scale actions to the actual parameter ranges
        temp_action_scaled = self._scale_action(temp_action_tanh, self.temp_min, self.temp_max)
        len_action_scaled = self._scale_action(len_action_tanh, self.len_min, self.len_max)

        # Clip actions to be within valid ranges and convert to appropriate types
        temp_action_clipped = torch.clamp(temp_action_scaled, self.temp_min, self.temp_max).item() # Float
        len_action_clipped = int(torch.round(torch.clamp(len_action_scaled, self.len_min, self.len_max)).item()) # Int

        # Return scaled/clipped actions and the log probability of the action *in tanh space*
        return temp_action_clipped, len_action_clipped, log_prob.item()

    def store_reward_and_done(self, reward, done=False):
        """Store reward and done signal for the last action taken."""
        # Should be called *after* select_action for the corresponding step
        if self.memory_size == 0 or self.memory_size != len(self.memory['rewards']) + 1:
            logger.warning("PPO store_reward_and_done called inconsistently with select_action.")
            # Decide how to handle: drop reward? raise error?
            return # Silently ignore for now

        # Ensure reward and done are tensors on CPU
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32).cpu()
        done_tensor = torch.as_tensor(done, dtype=torch.float32).cpu() # Use float for easier calculations

        self.memory['rewards'].append(reward_tensor)
        self.memory['dones'].append(done_tensor)

    def _calculate_gae(self, next_value):
        """Calculate Generalized Advantage Estimation (GAE)."""
        # Check if memory contains necessary components
        required_keys = ['rewards', 'dones', 'values']
        if not all(key in self.memory and self.memory[key] for key in required_keys):
            logger.warning("Cannot calculate GAE: Missing rewards, dones, or values in memory.")
            return torch.tensor([]), torch.tensor([])

        # Convert memory lists to tensors on the correct device
        rewards = torch.stack(self.memory['rewards']).to(self.device)
        dones = torch.stack(self.memory['dones']).to(self.device)
        values = torch.stack(self.memory['values']).to(self.device)

        # Ensure next_value is a tensor on the correct device
        if not isinstance(next_value, torch.Tensor):
             next_value_tensor = torch.tensor([next_value], dtype=torch.float32, device=self.device)
        else:
             next_value_tensor = next_value.to(self.device)
        # Ensure next_value is treated as a scalar estimate for V(s_T)
        if next_value_tensor.ndim > 0:
             next_value_tensor = next_value_tensor.squeeze() # Make scalar if needed


        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        num_steps = len(rewards)

        # Iterate backwards through the trajectory
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                # If this is the last step (s_T-1), V(s_T) is estimated by next_value
                next_non_terminal = 1.0 - dones[t] # Is s_T terminal? (Here, done applies to s_T-1 -> s_T transition)
                next_value_t = next_value_tensor # V(s_T)
            else:
                # If not the last step, V(s_t+1) is estimated by the critic's output for s_t+1
                next_non_terminal = 1.0 - dones[t] # Is s_t+1 terminal?
                next_value_t = values[t + 1]       # V(s_t+1)

            # Ensure value estimates are scalar
            if next_value_t.ndim > 0: next_value_t = next_value_t.squeeze()
            current_value_t = values[t]
            if current_value_t.ndim > 0: current_value_t = current_value_t.squeeze()


            # Calculate TD error (delta)
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - current_value_t
            # Calculate GAE advantage A(s_t, a_t)
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

        # Calculate returns (targets for the critic): Return(t) = A(t) + V(s_t)
        returns = advantages + values
        return advantages, returns

    def learn(self):
        """Update policy and critic networks using collected experiences."""
        MIN_BATCH_SIZE = 1 # Allow learning from a single feedback experience
        if self.memory_size < MIN_BATCH_SIZE:
            # logger.info(f"Not enough experiences for PPO learning ({self.memory_size} < {MIN_BATCH_SIZE})") # Can be noisy
            return 0.0, 0.0

        actor_loss_val, critic_loss_val = 0.0, 0.0

        # +++ PRINT: Check memory content before processing +++
        print(f">>> PPO LEARN: Memory size = {self.memory_size}")
        for key, val_list in self.memory.items():
            if val_list:
                # Ensure item is a tensor before checking shape
                item = val_list[0]
                shape_val = item.shape if isinstance(item, torch.Tensor) else item
                print(f"  Memory['{key}'][0] type: {type(item)}, shape/val: {shape_val}")
            else:
                print(f"  Memory['{key}'] is empty!")
        # +++ END PRINT +++

        with torch.no_grad():
            if not self.memory['states']:
                 logger.error("PPO learn called with empty states memory after size check.")
                 print(">>> PPO LEARN: ERROR - Empty states memory!") # DEBUG PRINT
                 return 0.0, 0.0
            # Use the state corresponding to the *last* value estimate in memory for V(s_T) calculation
            last_state = self.memory['states'][-1].unsqueeze(0).to(self.device)
            last_value = self.critic(last_state).squeeze()
            if last_value.ndim > 0:
                 last_value = last_value.squeeze() # Ensure scalar

            # +++ PRINT: Check last_value after potential squeeze +++
            print(f">>> PPO LEARN: last_value shape after squeeze: {last_value.shape}, value: {last_value.item() if last_value.numel() == 1 else last_value}")
            # +++ END PRINT +++

            advantages, returns = self._calculate_gae(last_value)

            # +++ PRINT: Check GAE outputs +++
            print(f">>> PPO LEARN: GAE calculated.")
            print(f"  advantages shape: {advantages.shape}, values: {advantages}")
            print(f"  returns shape: {returns.shape}, values: {returns}")
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                print(">>> PPO LEARN: ERROR - NaN/Inf detected in advantages!")
            if torch.isnan(returns).any() or torch.isinf(returns).any():
                print(">>> PPO LEARN: ERROR - NaN/Inf detected in returns!")
            # +++ END PRINT +++


            if advantages.numel() == 0 or returns.numel() == 0:
                logger.error("GAE calculation failed, skipping learn step")
                print(">>> PPO LEARN: ERROR - GAE calculation failed (empty tensors).") # DEBUG PRINT
                self.clear_memory()
                return 0.0, 0.0

        # Get old states, actions, and log probs from memory
        # Convert lists of tensors to stacked tensors
        old_states = torch.stack(self.memory['states']).to(self.device).detach()
        old_actions = torch.stack(self.memory['actions']).to(self.device).detach()
        old_logprobs = torch.stack(self.memory['logprobs']).to(self.device).detach()

        # +++ PRINT: Check tensors before loop +++
        print(f">>> PPO LEARN: Tensors prepared for loop.")
        print(f"  old_states shape: {old_states.shape}, has NaN: {torch.isnan(old_states).any()}")
        print(f"  old_actions shape: {old_actions.shape}, has NaN: {torch.isnan(old_actions).any()}")
        print(f"  old_logprobs shape: {old_logprobs.shape}, has NaN: {torch.isnan(old_logprobs).any()}")
        # +++ END PRINT +++


        # Normalize advantages across the batch (optional, but common)
        if advantages.numel() > 1: # Avoid normalizing if only one element (std dev is 0)
             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # If batch size is 1, advantage might be best left unnormalized or set to 0? Let's leave it.
        advantages = advantages.detach()
        returns = returns.detach()

        # +++ PRINT: Check normalized advantages and returns +++
        print(f">>> PPO LEARN: Post normalization/detach.")
        print(f"  advantages shape: {advantages.shape}, values: {advantages}, has NaN: {torch.isnan(advantages).any()}")
        print(f"  returns shape: {returns.shape}, values: {returns}, has NaN: {torch.isnan(returns).any()}")
        # +++ END PRINT +++


        # PPO update for K epochs - iterate over the collected batch multiple times
        print(f">>> PPO LEARN: Starting update loop (K_epochs={self.K_epochs})") # DEBUG PRINT
        for i in range(self.K_epochs):
            print(f"--- PPO Epoch {i+1}/{self.K_epochs} ---") # DEBUG PRINT

            # Get new action distributions from the *current* policy (self.policy)
            # +++ PRINT: Check actor input +++
            print(f"  Actor Input (old_states) shape: {old_states.shape}, has NaN: {torch.isnan(old_states).any()}")
            # +++ END PRINT +++
            temp_mean_tanh, temp_std, len_mean_tanh, len_std = self.policy(old_states)

            # +++ PRINT: Check actor outputs +++
            print(f"  Actor Output temp_mean_tanh: {temp_mean_tanh}, has NaN: {torch.isnan(temp_mean_tanh).any()}")
            print(f"  Actor Output temp_std: {temp_std}, has NaN: {torch.isnan(temp_std).any()}")
            print(f"  Actor Output len_mean_tanh: {len_mean_tanh}, has NaN: {torch.isnan(len_mean_tanh).any()}")
            print(f"  Actor Output len_std: {len_std}, has NaN: {torch.isnan(len_std).any()}")
            # Check for NaN right before distribution creation
            if torch.isnan(temp_mean_tanh).any() or torch.isnan(temp_std).any() or \
               torch.isnan(len_mean_tanh).any() or torch.isnan(len_std).any():
                print(">>> PPO LEARN: ERROR - NaN detected in Actor network output BEFORE creating distributions!")
                self.clear_memory() # Clear bad memory
                return 0.0, 0.0 # Exit learn function if NaN detected
            # +++ END PRINT +++

            # Create distributions
            temp_dist = Normal(temp_mean_tanh, temp_std)
            len_dist = Normal(len_mean_tanh, len_std)

            # Evaluate log probabilities of the *old actions* under the *new policy*
            temp_logprobs_now = temp_dist.log_prob(old_actions[:, 0].unsqueeze(1))
            len_logprobs_now = len_dist.log_prob(old_actions[:, 1].unsqueeze(1))
            logprobs_now = temp_logprobs_now + len_logprobs_now
            # Ensure shapes match old logprobs (handle potential extra dimensions)
            if logprobs_now.shape != old_logprobs.shape:
                 try:
                      logprobs_now = logprobs_now.reshape(old_logprobs.shape)
                      print(f"  Reshaped logprobs_now to {logprobs_now.shape} to match old_logprobs {old_logprobs.shape}")
                 except RuntimeError as reshape_err:
                      logger.error(f"PPO logprob shape mismatch! New: {logprobs_now.shape}, Old: {old_logprobs.shape}. Cannot reshape: {reshape_err}")
                      print(f">>> PPO LEARN: ERROR - Logprob shape mismatch! New: {logprobs_now.shape}, Old: {old_logprobs.shape}. Cannot reshape.")
                      continue # Skip epoch


            # Calculate distribution entropy (for encouraging exploration)
            entropy = temp_dist.entropy().mean() + len_dist.entropy().mean()

            # Get state value estimates from the *current* critic
            state_values = self.critic(old_states).squeeze()

            # +++ FIX SHAPE MISMATCH & PRINT +++
            print(f"  Critic Output (state_values) shape before reshape: {state_values.shape}")
            print(f"  Returns shape before reshape: {returns.shape}")
            # Ensure returns is compatible for mse_loss, should ideally match state_values shape
            if state_values.shape != returns.shape:
                try:
                    # Attempt to reshape returns to match state_values if possible
                    returns_reshaped = returns.reshape(state_values.shape)
                    print(f"  Reshaping returns to match state_values shape: {state_values.shape}")
                except RuntimeError as reshape_err:
                    print(f">>> PPO LEARN: ERROR - Cannot reshape returns {returns.shape} to match state_values {state_values.shape}. Error: {reshape_err}")
                    logger.error(f"PPO shape mismatch! State Values: {state_values.shape}, Returns: {returns.shape}. Cannot reshape.")
                    continue # Skip epoch
            else:
                returns_reshaped = returns # Shapes already match
            print(f"  state_values shape for loss: {state_values.shape}")
            print(f"  returns_reshaped shape for loss: {returns_reshaped.shape}")
            # +++ END FIX SHAPE MISMATCH & PRINT +++

            # --- PPO Objective Calculation ---
            # Ratio of new policy probability to old policy probability: r(t) = pi(a|s) / pi_old(a|s)
            ratios = torch.exp(logprobs_now - old_logprobs)

            # Clipped Surrogate Objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() # Maximize objective = Minimize negative objective

            # Critic Loss (Value Function Loss) - Use potentially reshaped returns
            critic_loss = F.mse_loss(state_values, returns_reshaped)

            # Total Loss (Actor + Critic + Entropy Bonus)
            # We maximize entropy, so subtract entropy * coefficient from loss
            total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy # Common to weight critic loss

            # +++ PRINT LOSSES +++
            print(f"  Epoch {i+1} Losses: actor={actor_loss.item():.4f}, critic={critic_loss.item():.4f}, entropy={entropy.item():.4f}, total={total_loss.item():.4f}")
            if torch.isnan(actor_loss) or torch.isnan(critic_loss) or torch.isnan(entropy):
                 print(f">>> PPO LEARN: ERROR - NaN detected in calculated losses!")
                 self.clear_memory() # Clear bad memory
                 return 0.0, 0.0 # Exit learn function
            # +++ END PRINT +++

            # --- Optimization ---
            # Optimize Actor
            self.optimizer_actor.zero_grad()
            # Backpropagate actor loss component (or total loss, depending on architecture)
            # Standard PPO often separates actor and critic updates slightly differently,
            # but optimizing total loss with separate optimizers can work.
            # Let's stick to optimizing based on the component losses for clarity.
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer_actor.step()

            # Optimize Critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()

            # Store loss values from the last epoch for reporting
            if i == self.K_epochs - 1:
                actor_loss_val = actor_loss.item()
                critic_loss_val = critic_loss.item()

        # Update the old policy network weights to the current policy weights
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear memory after finishing learning epochs for this batch
        self.clear_memory()
        print(f">>> PPO LEARN: Finished update loop. Final losses: Actor={actor_loss_val:.4f}, Critic={critic_loss_val:.4f}") # DEBUG PRINT
        return actor_loss_val, critic_loss_val

    def clear_memory(self):
        """Clear the memory buffer."""
        self.memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': [], 'values': []}
        self.memory_size = 0

    def save_model(self):
        """Save model (actor, critic, optimizers) to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_SAVE_DIR, f"{self.name}_{timestamp}.pt")

        try:
            torch.save({
                'policy': self.policy.state_dict(), # Current policy
                'policy_old': self.policy_old.state_dict(), # Old policy (might be redundant if always updated)
                'critic': self.critic.state_dict(),
                'optimizer_actor': self.optimizer_actor.state_dict(),
                'optimizer_critic': self.optimizer_critic.state_dict(),
                # Save relevant config params used by this agent
                'state_dim': self.policy.shared_layers[0].in_features, # Get state dim from layer
                'temp_range': (self.temp_min, self.temp_max),
                'len_range': (self.len_min, self.len_max)
            }, model_path)
            logger.info(f"Saved {self.name} model to {model_path}")
            return model_path
        except Exception as e:
             logger.error(f"Error saving {self.name} model to {model_path}: {e}", exc_info=True)
             return None

    def load_model(self, model_path):
        """Load model (actor, critic, optimizers) from disk."""
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found")
            return False

        try:
             # Load checkpoint onto the correct device
            checkpoint = torch.load(model_path, map_location=self.device)

            # --- Dimension Check (Optional but recommended) ---
            saved_state_dim = checkpoint.get('state_dim')
            current_state_dim = self.policy.shared_layers[0].in_features
            if saved_state_dim is not None and saved_state_dim != current_state_dim:
                 logger.error(f"State dimension mismatch loading PPO model {model_path}. Expected {current_state_dim}, got {saved_state_dim}. Load aborted.")
                 return False
            # --- End Dimension Check ---


            self.policy.load_state_dict(checkpoint['policy'])
            self.policy_old.load_state_dict(checkpoint['policy_old']) # Load old policy too
            self.critic.load_state_dict(checkpoint['critic'])
            self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
            self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])

            # Load ranges if present in checkpoint
            if 'temp_range' in checkpoint:
                self.temp_min, self.temp_max = checkpoint['temp_range']
            if 'len_range' in checkpoint:
                self.len_min, self.len_max = checkpoint['len_range']

            # Ensure models are on the correct device after loading
            self.policy.to(self.device)
            self.policy_old.to(self.device)
            self.critic.to(self.device)

            logger.info(f"Loaded {self.name} model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading PPO model from {model_path}: {str(e)}", exc_info=True)
            return False