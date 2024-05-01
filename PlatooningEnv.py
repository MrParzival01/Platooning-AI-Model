import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci
from sumolib import checkBinary


# Global Variables
TOTAL_STEPS = 3000

class PlatooningEnv(gym.Env):
    def __init__(self):
        super(PlatooningEnv, self).__init__()

        self.STEPS = 0
        self.total_reward = 0

        self.headway_details1 = []
        self.headway_details2 = []
        self.rewards = []
        # Define action space: 0 - accelerate, 1 - decelerate, 2 - maintain speed
        self.action_space = spaces.Discrete(3)

        # Define observation space: [follower_speed, headway_distance]
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([50, 200]), dtype=np.float32)

        self.sumo_binary = checkBinary('sumo')
        traci.start([self.sumo_binary, "-c", "./maps/singlelane/singlelane.sumocfg", "--tripinfo-output", "tripinfo.xml"])

        for _ in range(20):
            traci.simulationStep()
            self.STEPS += 1

        # Initialize vehicles
        self.initialize_vehicles()

    def initialize_vehicles(self):
        # Get vehicle IDs
        vehicles = traci.vehicle.getIDList()

        # Sort vehicles based on their lane position (x coordinate)
        vehicles_sorted_by_x = sorted(vehicles, key=lambda veh: traci.vehicle.getPosition(veh)[0], reverse=True)

        # Assign leader, first follower, and second follower based on their positions
        self.leader = vehicles_sorted_by_x[0]
        self.first_follower = vehicles_sorted_by_x[1]
        self.second_follower = vehicles_sorted_by_x[2]

        # Set initial speeds
        traci.vehicle.setSpeed(self.leader, 10)
        traci.vehicle.setSpeed(self.first_follower, 15)
        traci.vehicle.setSpeed(self.second_follower, 15)

    def calculate_distance(self, leader_id, follower_id):
        leader_pos = traci.vehicle.getPosition(leader_id)
        follower_pos = traci.vehicle.getPosition(follower_id)

        distance = ((leader_pos[0] - follower_pos[0])**2 + (leader_pos[1] - follower_pos[1])**2)**0.5
        return distance

    def step(self, action):
        self.platoon_joining(action)

        self.adjust_leader_speed()

        traci.simulationStep()
        self.STEPS += 1
        
        reward = self.compute_reward()
        observation = self.update_observation()
        done = self.STEPS >= TOTAL_STEPS
        truncated = False
        self.total_reward += reward

        info = {}

        return observation, reward, done, truncated, info

    def adjust_leader_speed(self):
        # Desired headway distance between the leader and the first follower
        desired_headway = 15

        current_headway = self.calculate_distance(self.leader, self.first_follower)

        if current_headway < desired_headway:
            # If too close, decrease leader's speed
            new_speed_leader = max(traci.vehicle.getSpeed(self.leader) + 1, 15)  
        elif current_headway > desired_headway:
            # If too far, increase leader's speed, but set a maximum limit
            new_speed_leader = min(traci.vehicle.getSpeed(self.leader) - 1, 5)  
        else:
            # Maintain current speed if the distance is within an acceptable range
            new_speed_leader = traci.vehicle.getSpeed(self.leader)
        
        traci.vehicle.setSpeed(self.leader, new_speed_leader)

    def compute_reward(self):
        reward = 0

        if 0 < self.current_headway_first <= 25:
            reward += 10  # Positive reward for maintaining optimal distance
        else:
            reward -= 10  # Negative reward for being too close or too far
        return reward

    def update_observation(self):
        # Observation now includes details for both sets of followers
        current_speed_first_follower = traci.vehicle.getSpeed(self.first_follower)
        self.current_headway_first = self.calculate_distance(self.leader, self.first_follower)
        self.current_headway_second = self.calculate_distance(self.first_follower, self.second_follower)
        
        self.headway_details1.append(self.current_headway_first)
        self.headway_details2.append(self.current_headway_second)
        
        # Form the observation
        observation = np.array([current_speed_first_follower, self.current_headway_first], dtype=np.float32)
        return observation

    def reset(self, seed=None):
        self.rewards.append(self.total_reward)
        print(f"Total reward gained in the last episode: {self.total_reward}")
        self.STEPS = 0
        self.total_reward = 0
        traci.close()
        traci.start([self.sumo_binary, "-c", "./maps/singlelane/singlelane.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        for _ in range(20):
            traci.simulationStep()
            self.STEPS += 1
        self.initialize_vehicles()
        observation = self.update_observation()
        return observation

    def platoon_joining(self, action):

        if action == 0:
            new_speed_first_follower = traci.vehicle.getSpeed(self.first_follower) + 5
        elif action == 1:
            new_speed_first_follower = traci.vehicle.getSpeed(self.first_follower) - 5
        else:  # Maintain speed
            new_speed_first_follower = traci.vehicle.getSpeed(self.first_follower)
        
        traci.vehicle.setSpeed(self.first_follower, new_speed_first_follower)

        # Logic for adjusting the second follower's speed, aiming for a dynamic desired distance
        dynamic_desired_distance = 10 + 0.1 * traci.vehicle.getSpeed(self.first_follower)
        current_distance = self.calculate_distance(self.first_follower, self.second_follower)
        
        # Adjust the second follower's speed based on the dynamic desired distance
        if current_distance > dynamic_desired_distance + 5:
            new_speed_second_follower = traci.vehicle.getSpeed(self.second_follower) + 1
        elif current_distance < dynamic_desired_distance - 5:
            new_speed_second_follower = max(traci.vehicle.getSpeed(self.second_follower) - 1, 0)
        else:  # Slight adjustment based on the first follower's action
            if action == 0:  # Accelerate
                new_speed_second_follower = min(traci.vehicle.getSpeed(self.second_follower) + 0.5, traci.vehicle.getSpeed(self.first_follower))
            elif action == 1:  # Decelerate
                new_speed_second_follower = max(traci.vehicle.getSpeed(self.second_follower) - 0.5, 0)
            else:
                new_speed_second_follower = traci.vehicle.getSpeed(self.second_follower)

        # Update the speed of the second follower
        traci.vehicle.setSpeed(self.second_follower, new_speed_second_follower)

    def render(self, mode='human'):
        pass

    def close(self):
        traci.close()
