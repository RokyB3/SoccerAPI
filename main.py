from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import random
import math

app = FastAPI()

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

def random_vector2d(min_x=-52.5, max_x=52.5, min_y=-34.0, max_y=34.0):
    return {
        "x": random.uniform(min_x, max_x),
        "y": random.uniform(min_y, max_y)
    }

def random_angle():
    return random.uniform(-180.0, 180.0)

def get_random_action(state: WorldState) -> Action:
    actions = []
    
    # basic movement and ball interaction
    if state.self.isKickable:
        # when we can kick, prioritize kick actions
        actions.extend([
            lambda: Action(type="Body_SmartKick", params={
                "target_point": random_vector2d(),
                "first_speed": random.uniform(0.5, 3.0),
                "first_speed_thr": 0.5,
                "max_step": random.randint(1, 3)
            }),
            lambda: Action(type="Body_KickOneStep", params={
                "target_point": random_vector2d(),
                "first_speed": random.uniform(0.5, 3.0),
                "force_mode": random.choice([True, False])
            }),
            lambda: Action(type="Body_StopBall", params={}),
            lambda: Action(type="Body_HoldBall2008", params={}),
        ])
    else:
        # when we can't kick, prioritize movement and interception
        actions.extend([
            lambda: Action(type="Body_GoToPoint", params={
                "point": random_vector2d(),
                "dist_thr": 0.5,
                "max_dash_power": random.uniform(30.0, 100.0),
                "dash_speed": -1.0,
                "cycle": 100,
                "save_recovery": True,
                "dir_thr": 15.0,
                "omni_dash_dist_thr": 1.0,
                "use_back_dash": True
            }),
            lambda: Action(type="Body_Intercept2018", params={
                "save_recovery": True
            }),
            lambda: Action(type="Body_GoToPointDodge", params={
                "point": random_vector2d(),
                "dash_power": random.uniform(30.0, 100.0)
            })
        ])

    # always available actions
    actions.extend([
        lambda: Action(type="Body_TurnToAngle", params={
            "angle": random_angle()
        }),
        lambda: Action(type="Body_TurnToPoint", params={
            "point": random_vector2d(),
            "cycle": random.randint(1, 3)
        }),
        lambda: Action(type="Body_TurnToBall", params={
            "cycle": random.randint(1, 3)
        }),
        lambda: Action(type="Neck_TurnToPoint", params={
            "point": random_vector2d()
        }),
        lambda: Action(type="Neck_TurnToBall", params={}),
        lambda: Action(type="Neck_ScanField", params={}),
        lambda: Action(type="View_Wide", params={}),
        lambda: Action(type="View_Normal", params={}),
        lambda: Action(type="Body_Dribble2008", params={
            "target_point": random_vector2d(),
            "dist_thr": random.uniform(0.5, 2.0),
            "dash_power": random.uniform(30.0, 100.0),
            "dash_count": random.randint(5, 15),
            "dodge": True
        }),
        lambda: Action(type="Body_TackleToPoint", params={
            "point": random_vector2d(),
            "min_prob": random.uniform(0.3, 0.8),
            "min_speed": random.uniform(0.0, 1.0)
        })
    ])

    # weight selection based on state
    weights = [1.0] * len(actions)
    
    # increase weights for certain actions based on context
    if state.self.isKickable:
        # boost kick-related actions when ball is kickable
        weights[:4] = [5.0] * 4
    elif state.ball.posCount < 3:  # if we've seen ball recently
        # boost movement towards ball
        weights[4:7] = [3.0] * 3

    # normalize weights
    total = sum(weights)
    weights = [w/total for w in weights]

    # select and execute action generator
    action_gen = random.choices(actions, weights=weights)[0]
    print(action_gen())
    return action_gen()

@app.post("/act")
async def act(state: WorldState) -> Action:
    return get_random_action(state)