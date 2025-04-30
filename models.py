import torch
import torch.nn as nn
import torch.nn.functional as F
import collections


map_minx = -55
map_maxx = 55
map_miny = -35
map_maxy = 35

# pre-define scale factors for continuous actions
CONTINUOUS_ACTION_SCALE = {
    "turn_to_angle": [0, 360],

    "body_turn_to_point_x": [map_minx, map_maxx], 
    "body_turn_to_point_y": [map_miny, map_maxy],

    "tackle_to_point_x": [map_minx, map_maxx], 
    "tackle_to_point_y": [map_miny, map_maxy], 

    "neck_turn_to_point_x": [map_minx, map_maxx],
    "neck_turn_to_point_y": [map_miny, map_maxy], 

    "view_change_width": [0, 180],

    "bhv_before_kickoff_point_x": [map_minx, map_maxx], 
    "bhv_before_kickoff_point_y": [map_miny, map_maxy], 

    "body_dribble_x": [map_minx, map_maxx],
    "body_dribble_y": [map_miny, map_maxy],
    "body_dribble_dist_thr": [0, 1], # check
    "body_dribble_dash_power": [0, 50], # check

    "body_go_to_point_dodge_x": [map_minx, map_maxx],
    "body_go_to_point_dodge_y": [map_miny, map_maxy], 
    "body_go_to_point_dodge_dash_power": [0, 50],

    "body_go_to_point_x": [map_minx, map_maxx], 
    "body_go_to_point_y": [map_miny, map_maxy], 
    "body_go_to_point_dist_thr": [0, 1],
    "body_go_to_point_dash_power": [0, 50],
    "body_go_to_point_dash_speed": [0, 50], # check
    
    "body_kick_one_step_x": [map_minx, map_maxx], 
    "body_kick_one_step_y": [map_miny, map_maxy], 
    "body_kick_one_step_first_speed": [0, 50],

    "body_kick_to_relative_target_dist": [0, map_maxx],
    "body_kick_to_relative_angle_deg": [0, 360], # check (negatives?)

    "body_smart_kick_x": [map_minx, map_maxx],
    "body_smart_kick_y": [map_miny, map_maxy],
    "body_smart_kick_first_speed": [0, 50],
    "body_smart_kick_first_speed_thr": [0, 50],
}


# define discrete actions as indices
DISCRETE_ACTIONS = [
    "Body_TurnToAngle", # 1 continuous parameter count
    "Body_TurnToPoint", # 2
    "Body_TurnToBall", # 0
    "Body_TackleToPoint", # 2
    "Neck_TurnToPoint", # 2
    "Neck_TurnToBall", # 0
    "View_Wide", # 0
    "View_Normal", # 0
    "View_ChangeWidth", # 1
    "Bhv_BeforeKickoff", # 2
    "Bhv_Emergency", # 0
    "Bhv_ScanField", # 0
    "Body_AdvanceBall2009", # 0
    "Body_ClearBall2009", # 0
    "Body_Dribble2008", # 4
    "Body_GoToPointDodge", # 3
    "Body_GoToPoint", # 5
    "Body_HoldBall2008", # 0
    "Body_Intercept2018", # 0
    "Body_KickOneStep", # 3
    "Body_KickToRelative", # 2
    "Body_Pass", # 0
    "Body_SmartKick", # 4
]

CONTINUOUS_ACTIONS = [
    # discrete 0 :
    "turn_to_angle", # index: 0

    # 1 :
    "body_turn_to_point_x", # 1
    "body_turn_to_point_y", # 2

    # 3 :
    "tackle_to_point_x", # 3
    "tackle_to_point_y", # 4

    # 4 :
    "neck_turn_to_point_x", # 5
    "neck_turn_to_point_y", # 6

    # 8 :
    "view_change_width", # 7

    # 9 :
    "bhv_before_kickoff_point_x", # 8 
    "bhv_before_kickoff_point_y", # 9

    # 14 :
    "body_dribble_x", # 10
    "body_dribble_y", # 11
    "body_dribble_dist_thr", # 12
    "body_dribble_dash_power", # 13

    # 15 :
    "body_go_to_point_dodge_x", # 14
    "body_go_to_point_dodge_y", # 15
    "body_go_to_point_dodge_dash_power", # 16

    # 16 :
    "body_go_to_point_x", # 17
    "body_go_to_point_y", # 18
    "body_go_to_point_dist_thr", # 19
    "body_go_to_point_dash_power", # 20
    "body_go_to_point_dash_speed", # 21
    
    # 19 :
    "body_kick_one_step_x", # 22
    "body_kick_one_step_y", # 23
    "body_kick_one_step_first_speed", # 24

    # 20 :
    "body_kick_to_relative_target_dist", # 25
    "body_kick_to_relative_angle_deg", # 26

    # 22 :
    "body_smart_kick_x", # 27
    "body_smart_kick_y", # 28
    "body_smart_kick_first_speed", # 29
    "body_smart_kick_first_speed_thr", # 30
]

# Masks from discrete action to continuous actions
MASKS = {
    0: [0], # turn_to_angle
    1: [1, 2], # body_turn_to_point_x, body_turn_to_point_y
    3: [3, 4], # tackle_to_point_x, tackle_to_point_y
    4: [5, 6], # neck_turn_to_point_x, neck_turn_to_point_y
    8: [7], # view_change_width
    9: [8, 9], # bhv_before_kickoff_point_x, bhv_before_kickoff_point_y
    14: [10, 11, 12, 13], # body_dribble_x, body_dribble_y, body_dribble_dist_thr, body_dribble_dash_power
    15: [14, 15, 16], # body_go_to_point_dodge_x, body_go_to_point_dodge_y, body_go_to_point_dodge_dash_power
    16: [17, 18, 19, 20, 21], # body_go_to_point_x, body_go_to_point_y, body_go_to_point_dist_thr, body_go_to_point_dash_power, body_go_to_point_dash_speed
    19: [22, 23, 24], # body_kick_one_step_x, body_kick_one_step_y, body_kick_one_step_first_speed
    20: [25, 26], # body_kick_to_relative_target_dist, body_kick_to_relative_angle_deg
    22: [27, 28, 29, 30], # body_smart_kick_x, body_smart_kick_y, body_smart_kick_first_speed, body_smart_kick_first_speed_thr
}

# define tensor masks based on these (all zeroes except for 1 at the indices)
TENSOR_MASKS = {}
for i in range(len(DISCRETE_ACTIONS)):
    mask = torch.zeros(len(CONTINUOUS_ACTIONS))
    if i in MASKS:
        for j in MASKS[i]:
            mask[j] = 1
    TENSOR_MASKS[i] = mask

print("TENSOR_MASKS:\n", TENSOR_MASKS)


class ReplayMemory():
    def __init__(self, capacity, agent_type="rl"):
        self.buffer = collections.deque()
        self.intermediary_buffer = {}
        self.capacity = capacity
        self.agent_type = agent_type

    def push_intermediary(self, state, action_d, action_c, agent_id):
        self.intermediary_buffer[agent_id] = (state, action_d, action_c)

    def push_final(self, new_state, reward, done, agent_id):
        if self.intermediary_buffer[agent_id] is not None:
            state, action_d, action_c = self.intermediary_buffer[agent_id]
            self.buffer.append((state, action_d, action_c, new_state, reward, done))
        # self.intermediary_buffer.clear()

        while len(self.buffer) > self.capacity:
            self.buffer.popleft()

    def push_imitation(self, state, action_d, action_c):
        self.buffer.append((state, action_d, action_c))
        while len(self.buffer) > self.capacity:
            self.buffer.popleft()
    
    def sample_imitation(self, n):
        num = min(n, len(self.buffer))
        indices = torch.randint(0, len(self.buffer), (num,))
        batch = []
        for i in indices:
            batch.append(self.buffer[i])
        states, action_ds, action_cs = zip(*batch)
        
        # stack
        states = torch.stack(states)
        action_ds = torch.stack(action_ds)
        action_cs = torch.stack(action_cs)
        
        return states, action_ds, action_cs
    
    def sample_rl(self, n):
        num = min(n, len(self.buffer))
        indices = torch.randint(0, len(self.buffer), (num,))
        batch = []
        for i in indices:
            batch.append(self.buffer[i])
        states, action_ds, action_cs, new_states, rewards, dones = zip(*batch)
        
        # stack
        states = torch.stack(states)
        action_ds = torch.stack(action_ds)
        action_cs = torch.stack(action_cs)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        
        return states, action_ds, action_cs, new_states, rewards, dones

    def clear_intermediary(self):
        self.intermediary_buffer.clear()


class HierarchicalSACPolicy(nn.Module):
    def __init__(self, state_dim, discrete_dim, continuous_dims, buffer=ReplayMemory(10000), hidden_dim=256, agent_id=0, agent_type="RL"):
        super(HierarchicalSACPolicy, self).__init__()
        self.state_dim = state_dim
        self.discrete_dim = discrete_dim
        self.continuous_dims = continuous_dims
        self.hidden_dim = hidden_dim

        self.agent_id = agent_id
        self.agent_type = agent_type

        self.memory = buffer

        self.discrete_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, discrete_dim)
        )

        self.continuous_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*continuous_dims)
        )

        self.q_value1 = nn.Sequential(
            nn.Linear(state_dim + continuous_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.q_value2 = nn.Sequential(
            nn.Linear(state_dim + continuous_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.q1_target = nn.Sequential(
            nn.Linear(state_dim + continuous_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2_target = nn.Sequential(
            nn.Linear(state_dim + continuous_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # self.q_values_optimizer = torch.optim.Adam(self.q_value.parameters(), lr=0.001)
        self.continuous_policy_optimizer = torch.optim.Adam(self.continuous_policy.parameters(), lr=0.001)
        self.discrete_policy_optimizer = torch.optim.Adam(self.discrete_policy.parameters(), lr=0.001)
        
        # init entropy tuning
        self.target_entropy_d = -torch.log(torch.tensor(1.0 / discrete_dim)).item()
        self.target_entropy_c = -torch.log(torch.tensor(1.0 / continuous_dims)).item()
        self.log_alpha_d = torch.zeros(1, requires_grad=True)
        self.log_alpha_c = torch.zeros(1, requires_grad=True)
        self.alpha_d_optimizer = torch.optim.Adam([self.log_alpha_d], lr=0.001)
        self.alpha_c_optimizer = torch.optim.Adam([self.log_alpha_c], lr=0.001)
        self.alpha_d = torch.exp(self.log_alpha_d)
        self.alpha_c = torch.exp(self.log_alpha_c)
        self.log_alpha_d.requires_grad = True
        self.log_alpha_c.requires_grad = True


    # in: state
    # out: pi_d (discrete action probabilities), mu_c (mean of continuous actions), sigma_c (std of continuous actions)
    def forward(self, state):
        # first, discrete action: softmax gives pi_d
        pi_d = F.softmax(self.discrete_policy(state), dim=-1)

        # mu_c and sigma_c for continuous action
        continuous_action_params = self.continuous_policy(state)
        mu_c = continuous_action_params[:, :self.continuous_dims]
        sigma_c = F.softplus(continuous_action_params[:, self.continuous_dims:])

        # squashed gaussian: tanh(mu_c + sigma_c * N(0, 1))
        return pi_d, mu_c, sigma_c
    
    # # in: state, discrete action (one-hot), continuous action
    # # out: Q value (estimate of expected return)
    # def get_q_value(self, state, continuous_action):
    #     q_input = torch.cat([state, continuous_action], dim=-1)
    #     return self.q_value(q_input)

    # in: state
    # out: discrete action (one-hot), continuous actions (sampled from Gaussian then scaled)
    def choose_action(self, state, prev_reward, done):
        # save experience using environmental feedback
        self.memory.push_final(state, prev_reward, done, self.agent_id)
        if done:
            return

        # get distributions
        pi_d, mu_c, sigma_c = self.forward(state)
        # sample discrete action categorically
        dist_d = torch.Categorical(logits=pi_d)
        action_d = dist_d.sample()
        # sample and re-scale continuous action from Gaussian
        dist_c = torch.distributions.Normal(mu_c, sigma_c)
        sample_c = dist_c.rsample()
        action_c = torch.tanh(sample_c)

        # types:
        # action_d: torch.tensor([x]) where x is the index of the action selected
        # action_c: torch.tensor([x1, x2, ...]) where x1, x2, ... are all continuous actions

        # scale continuous actions based on discrete action
        # multiply by (max-min)/2 then add (max+min)/2
        scaled_c = (action_c * (CONTINUOUS_ACTION_SCALE[CONTINUOUS_ACTIONS[action_d.item()]][1] - CONTINUOUS_ACTION_SCALE[CONTINUOUS_ACTIONS[action_d.item()]][0]) / 2)+ (CONTINUOUS_ACTION_SCALE[CONTINUOUS_ACTIONS[action_d.item()]][1] + CONTINUOUS_ACTION_SCALE[CONTINUOUS_ACTIONS[action_d.item()]][0]) / 2
        
        masked_c = scaled_c * TENSOR_MASKS[action_d.item()]

        self.memory.push_intermediary(state, action_d, masked_c, self.agent_id)

        return action_d, masked_c
    
    def log_probs(self, state, action_d, action_c):
        # get distributions
        pi_d, mu_c, sigma_c = self.forward(state)
        # sample discrete action categorically
        dist_d = torch.Categorical(logits=pi_d)
        log_prob_d = dist_d.log_prob(action_d)
        # sample and re-scale continuous action from Gaussian
        dist_c = torch.distributions.Normal(mu_c, sigma_c)
        sample_c = dist_c.rsample()
        log_prob_c = dist_c.log_prob(sample_c) - torch.log(1 - action_c.pow(2) + 1e-6)

        return log_prob_d, log_prob_c
        
    def train(self):
        if self.agent_type == "RL":
            self.train_rl()
        elif self.agent_type == "imitation":
            self.train_imitation()

        # clear memory
        self.memory.clear_intermediary()
    
    def train_rl(self):
        # sample from memory
        states, action_ds, action_cs, new_states, rewards, dones = self.memory.sample_rl(64)
        
        # get Q values for current state and next state
        q1 = self.q_value1(states, action_cs)
        q2 = self.q_value2(states, action_cs)

        q1_next = self.q1_target(new_states, action_cs)
        q2_next = self.q2_target(new_states, action_cs)

        # get log probs for discrete and continuous actions
        log_prob_d, log_prob_c = self.log_probs(states, action_ds, action_cs)
        log_prob_d = log_prob_d.gather(1, action_ds.unsqueeze(1)).squeeze(1)
        log_prob_c = log_prob_c.gather(1, action_cs.unsqueeze(1)).squeeze(1)
        
        # get target Q value
        target_q = rewards + (1 - dones) * 0.99 * torch.min(q1_next, q2_next)

        # compute Q loss
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        # update Q value networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        

        # update alpha
        alpha_d_loss = (self.log_alpha_d * (log_prob_d + self.target_entropy_d)).mean()
        alpha_c_loss = (self.log_alpha_c * (log_prob_c + self.target_entropy_c)).mean()
        self.alpha_d_optimizer.zero_grad()
        alpha_d_loss.backward()
        self.alpha_d_optimizer.step()
        self.alpha_c_optimizer.zero_grad()
        alpha_c_loss.backward()
        self.alpha_c_optimizer.step()
        # update target networks
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
        
        # update policy networks according to SAC
        policy_loss_d = (log_prob_d * (self.alpha_d * log_prob_d - torch.min(q1, q2))).mean()
        policy_loss_c = (log_prob_d * (self.alpha_c * log_prob_d * log_prob_c - torch.min(q1, q2))).mean()

        self.discrete_policy_optimizer.zero_grad()
        policy_loss_d.backward()
        self.discrete_policy_optimizer.step()
        self.continuous_policy_optimizer.zero_grad()
        policy_loss_c.backward()
        self.continuous_policy_optimizer.step()

    def train_imitation(self):

        num_epochs = 5
        # train with direct behavioural cloning
        for i in range(num_epochs):
            states, action_ds, action_cs = self.memory.sample_imitation(64)
            # ignore Q values: don't care about rewards whatsoever.

            # get log probs for discrete and continuous actions
            log_prob_d, log_prob_c = self.log_probs(states, action_ds, action_cs)
            log_prob_d = log_prob_d.gather(1, action_ds.unsqueeze(1)).squeeze(1)
            log_prob_c = log_prob_c.gather(1, action_cs.unsqueeze(1)).squeeze(1)

            # we want discrete_policy to output the same action as action_ds
            # and continuous_policy to output the same action as action_cs
            # so we can use cross entropy loss for discrete and MSE for continuous
    
            # discrete action loss
            discrete_action_loss = F.cross_entropy(self.discrete_policy(states), action_ds)
            # continuous action loss
            continuous_action_loss = F.mse_loss(self.continuous_policy(states), action_cs)
            # combine losses
            imitation_loss_d = discrete_action_loss + self.alpha_d * log_prob_d.mean()
            imitation_loss_c = continuous_action_loss + self.alpha_c * log_prob_c.mean()

            # propagate gradients
            self.discrete_policy_optimizer.zero_grad()
            imitation_loss_d.backward()
            self.discrete_policy_optimizer.step()

            self.continuous_policy_optimizer.zero_grad()
            imitation_loss_c.backward()
            self.continuous_policy_optimizer.step()

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        
    def clone_weights(self, agent_list):
        for other_agent in agent_list:
            if other_agent.agent_id != self.agent_id:
                self.discrete_policy.load_state_dict(other_agent.discrete_policy.state_dict())
                self.continuous_policy.load_state_dict(other_agent.continuous_policy.state_dict())
                self.q_value1.load_state_dict(other_agent.q_value1.state_dict())
                self.q_value2.load_state_dict(other_agent.q_value2.state_dict())
                self.q1_target.load_state_dict(other_agent.q1_target.state_dict())
                self.q2_target.load_state_dict(other_agent.q2_target.state_dict())