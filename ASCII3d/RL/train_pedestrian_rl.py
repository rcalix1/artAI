import random, json
import numpy as np

# Tiny tabular Q-learning experiment matching the ASCII city's cross street.
EPISODES = 15000
ALPHA = 0.18
GAMMA = 0.97
EPS_START = 1.0
EPS_END = 0.03
DT = 0.20
CAR_SPEED_MIN, CAR_SPEED_MAX = 5.0, 10.0
PED_SPEED = 3.5
CROSS_X = -20.0
START_Z = 45.0
END_Z = 65.0
ROAD_Z = 55.0

# Relative car x = car.x - crossing x. Car moves toward +x.
CAR_BINS = [-40,-25,-15,-10,-7,-5,-3,0,3,7,15,30,60]
PED_BINS = [45,48,51,53,55,57,59,62,65]
N_CAR = len(CAR_BINS)+1
N_PED = len(PED_BINS)+1
Q = np.zeros((N_CAR, N_PED, 2), dtype=np.float64) # 0 WAIT, 1 WALK

def state(car_x, ped_z):
    return (int(np.digitize(car_x-CROSS_X, CAR_BINS)),
            int(np.digitize(ped_z, PED_BINS)))

def collision(car_x, ped_z):
    return abs(car_x-CROSS_X) < 2.4 and abs(ped_z-ROAD_Z) < 2.2

wins = hits = 0
for ep in range(EPISODES):
    car_x = random.uniform(-90,-30)
    car_speed = random.uniform(CAR_SPEED_MIN,CAR_SPEED_MAX)
    ped_z = START_Z
    eps = EPS_END + (EPS_START-EPS_END)*max(0.0, 1.0-ep/(EPISODES*0.85))
    for _ in range(500):
        s = state(car_x,ped_z)
        if random.random() < eps:
            a = random.randrange(2)
        else:
            a = int(np.argmax(Q[s]))

        old_z = ped_z
        if a == 1:
            ped_z += PED_SPEED*DT
        car_x += car_speed*DT

        reward = -0.05 + (ped_z-old_z)*0.08
        done = False
        if collision(car_x,ped_z):
            reward = -100.0; done = True; hits += 1
        elif ped_z >= END_Z:
            reward = 100.0; done = True; wins += 1
        elif car_x > 75:
            # Another car will eventually arrive; wrap it as in the HTML world.
            car_x = -90
            car_speed = random.uniform(CAR_SPEED_MIN,CAR_SPEED_MAX)

        ns = state(car_x,ped_z)
        target = reward if done else reward + GAMMA*np.max(Q[ns])
        Q[s+(a,)] += ALPHA*(target-Q[s+(a,)])
        if done: break

policy = np.argmax(Q,axis=2).astype(int).tolist()
data = {
    'car_bins': CAR_BINS,
    'ped_bins': PED_BINS,
    'policy': policy,
    'actions': ['WAIT','WALK'],
    'episodes': EPISODES,
    'cross_x': CROSS_X,
    'start_z': START_Z,
    'end_z': END_Z,
    'road_z': ROAD_Z
}
with open('/mnt/data/pedestrian_policy.json','w') as f:
    json.dump(data,f,indent=2)
with open('/mnt/data/pedestrian_policy.js','w') as f:
    f.write('window.RL_POLICY = ' + json.dumps(data) + ';\n')
print('Training complete')
print('episodes:',EPISODES,'training successes:',wins,'collisions:',hits)
print('saved pedestrian_policy.json and pedestrian_policy.js')
