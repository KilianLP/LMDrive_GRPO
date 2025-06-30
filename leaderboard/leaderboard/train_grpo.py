import carla
import numpy as np
import random
import time
import os

from leaderboard.scenarios.route_indexer import RouteIndexer
from srunner.scenariomanager.route_scenario import RouteScenario
from srunner.tools.scenario_tools import remove_all_actors  # optional helper
from leaderboard.envs.sensor_interface import SensorInterface

from LAVIS.lavis.models.drive_models.drive_grpo import Blip2VicunaDrive


# === Settings ===
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
TM_PORT = 8000
TOWN = 'Town05'
ROUTE_FILE = 'langauto/benchmark_tiny.xml'
SCENARIO_FILE = 'scenarios/all_towns_traffic_scenarios.json'
REPETITIONS = 1
MAX_TIMESTEPS = 1000  # or however long each episode lasts


# === Connect to CARLA ===
client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(10.0)
traffic_manager = client.get_trafficmanager(TM_PORT)

world = client.load_world(TOWN)
settings = world.get_settings()
settings.fixed_delta_seconds = 0.05  # 20 Hz
settings.synchronous_mode = True
world.apply_settings(settings)
traffic_manager.set_synchronous_mode(True)


# === Load routes and scenarios ===
indexer = RouteIndexer(
    route_filename=ROUTE_FILE,
    scenario_file=SCENARIO_FILE,
    repetitions=REPETITIONS
)
indexer.prepare_next_entry()


# === Instantiate agent ===
agent = Blip2VicunaDrive()

def compute_custom_reward(ego_vehicle, sensor_data, world, route_waypoints=None):
    """
    Compute reward based on the ego vehicle's driving behavior.

    Args:
        ego_vehicle (carla.Vehicle): the autonomous car
        sensor_data (dict): dictionary with 'collision', 'lane_invasion', etc.
        world (carla.World): CARLA world object (to get map, traffic lights)
        route_waypoints (List[carla.Location], optional): planned route for guidance

    Returns:
        float: reward value
    """
    reward = 0.0

    # === 1. Collision Penalty ===
    collision_event = sensor_data.get('collision')
    if collision_event and len(collision_event) > 0:
        return -100.0

    # === 2. Lane Invasion Penalty ===
    lane_event = sensor_data.get('lane_invasion')
    if lane_event and len(lane_event.crossed_lane_markings) > 0:
        reward -= 5.0  # penalize crossing lines

    # === 3. Traffic Light or Stop Sign Violation ===
    tl = ego_vehicle.get_traffic_light()
    if tl is not None and tl.get_state() == carla.TrafficLightState.Red:
        if ego_vehicle.get_velocity().length() > 0.5:
            reward -= 10.0  # running a red light

    # You can also check for stop signs via map.get_waypoint(...).is_stop_sign
    map = world.get_map()
    waypoint = map.get_waypoint(ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    if waypoint.is_junction and "Stop" in waypoint.get_landmarks(10):
        if ego_vehicle.get_velocity().length() > 0.5:
            reward -= 5.0  # not stopping at a stop sign

    # === 4. Route Progress Reward ===
    if route_waypoints:
        current_loc = ego_vehicle.get_location()
        closest_wp = min(route_waypoints, key=lambda p: p.distance(current_loc))
        dist_to_route = current_loc.distance(closest_wp)
        reward += max(0, 5.0 - dist_to_route) * 0.1  # reward proximity to route center

        # Optionally: reward forward motion along the route
        vehicle_wp = map.get_waypoint(current_loc, project_to_road=True)
        idx = route_waypoints.index(closest_wp) if closest_wp in route_waypoints else 0
        forward_progress = idx / len(route_waypoints)
        reward += forward_progress * 2.0

    # === 5. Speed Incentive ===
    v = ego_vehicle.get_velocity()
    speed = np.linalg.norm([v.x, v.y, v.z])
    reward += speed * 0.05  # encourage some motion

    # === 6. Steering Smoothness Penalty ===
    control = ego_vehicle.get_control()
    reward -= 2.0 * abs(control.steer)

    return reward




# === Main loop over routes ===
while indexer.route_valid():
    route_config = indexer.next()

    # Setup scenario (spawns NPCs, ego vehicle, sensors)
    scenario = RouteScenario(route_config, ego_vehicle_num=1, debug_mode=False)
    ego_vehicle = scenario.ego_vehicles[0]
    sensor_interface = scenario.sensor_interface

    # Optional: init custom state tracker, logger, etc.

    # === Episode loop ===
    for t in range(MAX_TIMESTEPS):
        world.tick()

        # Retrieve observations
        sensor_data = sensor_interface.get_data()
        timestamp = world.get_snapshot().timestamp

        # Agent chooses action
        control = agent.run_step(sensor_data, timestamp)  # make sure your agent takes correct inputs

        # Apply control
        ego_vehicle.apply_control(control)

        # Reward and done logic (custom functions you must define)
        reward = compute_custom_reward(ego_vehicle, sensor_data)
        done = check_termination_conditions(ego_vehicle, sensor_data)

        # Optionally: store transition for training
        agent.store_transition(sensor_data, control, reward, done)

        if done:
            break

    # === Training step (optional) ===
    agent.update_policy()

    # Cleanup actors from this scenario
    scenario.remove_all_actors()

    # Optional: reset world state, clear memory, etc.

print("All scenarios completed.")
