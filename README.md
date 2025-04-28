# SoccerAPI

```
apt-get update
apt install python3.12-venv
```

```
cd SoccerAPI
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```
python manage.py runserver
```

Add rc/teams/base_teams/helios-base-support-v18/src/player/serialize_world.cpp to serialize the world model to json.
Add rc/teams/base_teams/helios-base-support-v18/src/player/send_post_request.cpp to send the json to the API.
Add rc/teams/base_teams/helios-base-support-v18/src/player/httplib.h to use the API.