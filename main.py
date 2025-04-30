from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import random
import math
import torch

# Import rl models
from models import HierarchicalSACPolicy, DISCRETE_ACTIONS, CONTINUOUS_ACTIONS, MASKS, CONTINUOUS_ACTION_SCALE, ReplayMemory

#     global policy
policy = []
buffer = ReplayMemory(10000)
load=False
num_agents=11
training_type="RL"


if training_type == "RL":
    
    policy = [HierarchicalSACPolicy(buffer=buffer, agent_id=i, agent_type="RL", state_dim=17, discrete_dim=13, continuous_dims=13) for i in range(num_agents)]
    if load:
        # Load the model if needed
        loaded_model = HierarchicalSACPolicy(buffer=buffer, agent_id=999, agent_type="RL", state_dim=17, discrete_dim=13, continuous_dims=13)
        loaded_model.load(f"rl_weights.pt")
        loaded_model.clone_weights(policy) # duplicates weights to all other agents
else:
    policy = [HierarchicalSACPolicy(buffer=buffer, agent_id=i, agent_type="imitation", state_dim=17, discrete_dim=13, continuous_dims=13) for i in range(num_agents)]
    if load:
        # Load the model if needed
        loaded_model = HierarchicalSACPolicy(buffer=buffer, agent_id=999, agent_type="imitation", state_dim=17, discrete_dim=13, continuous_dims=13)
        loaded_model.load(f"imitation_weights.pt")
        loaded_model.clone_weights(policy) # duplicates weights to all other agents
print("Policy initialized")

app = FastAPI()
# # stateful memory for each agent (by unum)
# agent_memories = {}

# class AgentMemory:
#     def __init__(self):
#         self.last_state = None
#         self.last_action_d = None
#         self.last_action_c = None
#         self.last_reward = 0.0
#         self.last_time = None

class Position(BaseModel):
    x: float
    y: float

class SelfState(BaseModel):
    isFrozen: bool
    tackleExpires: int
    unum: int
    posValid: bool
    isKickable: bool
    goalie: bool
    pos: Position
    faceValid: bool

class BallState(BaseModel):
    posCount: int
    velCount: int
    seenPosCount: int

class TimeState(BaseModel):
    current: int
    stopped: int

class WorldState(BaseModel):
    self: SelfState
    ball: BallState
    time: TimeState

class Action(BaseModel):
    type: str
    params: Dict[str, Any]

# def random_vector2d(min_x=-52.5, max_x=52.5, min_y=-34.0, max_y=34.0):
#     return {
#         "x": random.uniform(min_x, max_x),
#         "y": random.uniform(min_y, max_y)
#     }

# def random_angle():
#     return random.uniform(-180.0, 180.0)

# def get_random_action(state: WorldState) -> Action:
#     actions = []
    
#     # basic movement and ball interaction
#     if state.self.isKickable:
#         # when we can kick, prioritize kick actions
#         actions.extend([
#             lambda: Action(type="Body_SmartKick", params={
#                 "target_point": random_vector2d(),
#                 "first_speed": random.uniform(0.5, 3.0),
#                 "first_speed_thr": 0.5,
#                 "max_step": random.randint(1, 3)
#             }),
#             lambda: Action(type="Body_KickOneStep", params={
#                 "target_point": random_vector2d(),
#                 "first_speed": random.uniform(0.5, 3.0),
#                 "force_mode": random.choice([True, False])
#             }),
#             lambda: Action(type="Body_StopBall", params={}),
#             lambda: Action(type="Body_HoldBall2008", params={}),
#         ])
#     else:
#         # when we can't kick, prioritize movement and interception
#         actions.extend([
#             lambda: Action(type="Body_GoToPoint", params={
#                 "point": random_vector2d(),
#                 "dist_thr": 0.5,
#                 "max_dash_power": random.uniform(30.0, 100.0),
#                 "dash_speed": -1.0,
#                 "cycle": 100,
#                 "save_recovery": True,
#                 "dir_thr": 15.0,
#                 "omni_dash_dist_thr": 1.0,
#                 "use_back_dash": True
#             }),
#             lambda: Action(type="Body_Intercept2018", params={
#                 "save_recovery": True
#             }),
#             lambda: Action(type="Body_GoToPointDodge", params={
#                 "point": random_vector2d(),
#                 "dash_power": random.uniform(30.0, 100.0)
#             })
#         ])

#     # always available actions
#     actions.extend([
#         lambda: Action(type="Body_TurnToAngle", params={
#             "angle": random_angle()
#         }),
#         lambda: Action(type="Body_TurnToPoint", params={
#             "point": random_vector2d(),
#             "cycle": random.randint(1, 3)
#         }),
#         lambda: Action(type="Body_TurnToBall", params={
#             "cycle": random.randint(1, 3)
#         }),
#         lambda: Action(type="Neck_TurnToPoint", params={
#             "point": random_vector2d()
#         }),
#         lambda: Action(type="Neck_TurnToBall", params={}),
#         lambda: Action(type="Neck_ScanField", params={}),
#         lambda: Action(type="View_Wide", params={}),
#         lambda: Action(type="View_Normal", params={}),
#         lambda: Action(type="Body_Dribble2008", params={
#             "target_point": random_vector2d(),
#             "dist_thr": random.uniform(0.5, 2.0),
#             "dash_power": random.uniform(30.0, 100.0),
#             "dash_count": random.randint(5, 15),
#             "dodge": True
#         }),
#         lambda: Action(type="Body_TackleToPoint", params={
#             "point": random_vector2d(),
#             "min_prob": random.uniform(0.3, 0.8),
#             "min_speed": random.uniform(0.0, 1.0)
#         })
#     ])

#     # weight selection based on state
#     weights = [1.0] * len(actions)
    
#     # increase weights for certain actions based on context
#     if state.self.isKickable:
#         # boost kick-related actions when ball is kickable
#         weights[:4] = [5.0] * 4
#     elif state.ball.posCount < 3:  # if we've seen ball recently
#         # boost movement towards ball
#         weights[4:7] = [3.0] * 3

#     # normalize weights
#     total = sum(weights)
#     weights = [w/total for w in weights]

#     # select and execute action generator
#     action_gen = random.choices(actions, weights=weights)[0]
#     return action_gen()

# def initialize_server(load=False, num_agents=11, training_type="RL"):


def state_to_tensor(state: WorldState):
    # flatten state to tensor, must match model's expected input
    # this is a stub, you need to implement this properly
    arr = [
        float(state.self.isFrozen),
        float(state.self.tackleExpires),
        float(state.self.unum),
        float(state.self.posValid),
        float(state.self.isKickable),
        float(state.self.goalie),
        float(state.self.pos.x),
        float(state.self.pos.y),
        float(state.self.faceValid),
        float(state.ball.posCount),
        float(state.ball.velCount),
        float(state.ball.seenPosCount),
        float(state.time.current),
        float(state.time.stopped),
        # add more features as needed
    ]
    # pad to 17 dims if needed
    while len(arr) < 17:
        arr.append(0.0)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # shape (1, 17)

def reward_from_state(state: WorldState, prev_state: WorldState):
    # -1 per step, +100 if goal scored (dummy: you need to implement real goal detection)
    # for now, just -1 per step
    return -1.0

def action_to_api_action(action_d, action_c):
    # action_d: torch.tensor([idx])
    # action_c: torch.tensor([[c1, c2, ...]]) shape (1, N)
    idx = int(action_d.item())
    action_type = DISCRETE_ACTIONS[idx]
    params = {}

    # get which continuous params are relevant for this action
    mask = MASKS.get(idx, [])
    action_c = action_c.squeeze(0)  # shape (N,)

    # map continuous params to their names and scale
    for i, c_idx in enumerate(mask):
        param_name = CONTINUOUS_ACTIONS[c_idx]
        val = action_c[c_idx].item()

        # now map param_name to the expected json param
        # e.g. "body_turn_to_point_x" -> "point": {"x": ...}
        if "point_x" in param_name:
            base = param_name.rsplit("_x", 1)[0]
            if base not in params:
                params[base] = {}
            params[base]["x"] = val
        elif "point_y" in param_name:
            base = param_name.rsplit("_y", 1)[0]
            if base not in params:
                params[base] = {}
            params[base]["y"] = val
        elif "angle" in param_name:
            params["angle"] = val
        elif "dist_thr" in param_name:
            params["dist_thr"] = val
        elif "dash_power" in param_name:
            params["dash_power"] = val
        elif "first_speed" in param_name:
            params["first_speed"] = val
        elif "first_speed_thr" in param_name:
            params["first_speed_thr"] = val
        elif "max_step" in param_name:
            params["max_step"] = int(val)
        elif "cycle" in param_name:
            params["cycle"] = int(val)
        elif "min_prob" in param_name:
            params["min_prob"] = val
        elif "min_speed" in param_name:
            params["min_speed"] = val
        elif "target_dist" in param_name:
            params["target_dist"] = val
        elif "target_angle_deg" in param_name:
            params["target_angle_relative"] = val
        # add more as needed

    # for actions that expect a "point" param, merge x/y if present
    for k in list(params.keys()):
        if k.endswith("point"):
            if "x" in params[k] and "y" in params[k]:
                params["point"] = {"x": params[k]["x"], "y": params[k]["y"]}
                del params[k]

    # fill in required params for actions with no continuous params
    # (e.g. empty dict)
    if not params:
        params = {}

    return Action(type=action_type, params=params)

@app.post("/act")
async def act(state: WorldState, agent_id: int) -> Action:
    # unum = state.self.unum
    # if unum not in agent_memories:
    #     agent_memories[unum] = AgentMemory()
    # mem = agent_memories[unum]
    print(state)
    s = state_to_tensor(state)
    done = False
    # first_step = False

    """
    We can deal with making the done work right here I guess?
    
    """

    reward = reward_from_state(state)#, mem.last_state)
    

    # train at end of ep, otherwise choose action
    if done and agent_id == 0:
        policy[0].train()
        policy[0].clone_weights(policy) # duplicates weights to all other agents
    else:    
        with torch.no_grad():
            action_d, action_c = policy[agent_id].choose_action(s, reward, done)
    
    
        

    # mem.last_state = state
    # mem.last_action_d = action_d
    # mem.last_action_c = action_c
    # mem.last_reward = reward

    # policy.train()

    # convert RL action to API action
    return action_to_api_action(action_d, action_c)

# add endpoints to save/load model if needed
@app.post("/save")
async def save():
    if policy[0].agent_type == "RL":
        policy[0].save("rl_weights.pt")
    else:
        policy[0].save("imitation_weights.pt")
    return {"status": "saved"}


# @app.post("/load")
# async def load():
#     policy.load("policy.pt")
#     return {"status": "loaded"}