from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

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

@app.post("/act", response_model=Action)
async def act(state: WorldState):
    # TODO: insert RL agent logic here
    # for now, just a dummy action
    action = Action(
        type="Body_GoToPoint",
        params={
            "x": 0,
            "y": 0,
            "dist_thr": 0.5,
            "max_dash_power": 1.0
        }
    )
    return action