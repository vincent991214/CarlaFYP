"""
https://github.com/carla-simulator/carla/issues/3890
"""
import math
import time
import carla
import numpy as np

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

start_time = time.time()
d = 0.0


def get_player_car_distance(x, y):
    global d
    for target in world.get_actors():
        if target.attributes.get('role_name') == 'hero':
            player = target
            p_location = player.get_location()
            d = math.sqrt((x - p_location.x) ** 2 + (y - p_location.y) ** 2)
            break
    return d


def get_player_car_speed():
    for target in world.get_actors():
        if target.attributes.get('role_name') == 'hero':
            player = target
            velocity = player.get_velocity()
            transform = player.get_transform()
            break
    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return str(round(3.6 * speed, 2))


# https://blog.css8.cn/post/19738092.html
def get_player_car_yaw():
    for target in world.get_actors():
        if target.attributes.get('role_name') == 'hero':
            player = target
            velocity = player.get_velocity()
            transform = player.get_transform()
            break
    yaw = transform.rotation.yaw
    return str(yaw)


f = open("Delta_Distance.txt", "w")
f.write("DeltaD Time Yaw Speed" + "\n")

# f = open("WaypointInScenarioIII.txt", "w")
# f.write("Waypoint_x Waypoint_y" + "\n")

client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.load_world("Town07")

blueprint_library = world.get_blueprint_library()
amap = world.get_map()
sampling_resolution = 1
dao = GlobalRoutePlannerDAO(amap, sampling_resolution)
grp = GlobalRoutePlanner(dao)
grp.setup()

spawn_points = world.get_map().get_spawn_points()
start_point = carla.Location(x=-3, y=122, z=1)
end_point = carla.Location(x=-201, y=-227, z=1)
# There are other functions can be used to generate a route in GlobalRoutePlanner.
routine = grp.trace_route(start_point, end_point)

spectator = world.get_spectator()
spectator.set_transform(carla.Transform(start_point + carla
                                        .Location(x=5, y=5, z=150), carla
                                        .Rotation(yaw=0, pitch=-90)))

i = 0
for w in routine:
    if i % 10 == 0:
        world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=100000.0,
                                persistent_lines=True)
    else:
        world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                                color=carla.Color(r=0, g=0, b=255), life_time=100000.0,
                                persistent_lines=True)
    i += 1
    # print(w[0].lane_id, w[0].road_id, w[0].transform.location)
    # print(w[0].transform.location)
    # f.write(str(w[0].transform.location.x) + " ")
    # f.write(str(w[0].transform.location.y) + "\n")

# print(routine[111][0].transform.location)
world.debug.draw_string(routine[459][0].transform.location, 'O', draw_shadow=False,
                        color=carla.Color(r=0, g=255, b=0), life_time=100000.0,
                        persistent_lines=True)
world.debug.draw_string(routine[467][0].transform.location, 'O', draw_shadow=False,
                        color=carla.Color(r=0, g=255, b=0), life_time=100000.0,
                        persistent_lines=True)
# -10, 120 start
TempList = []
DeltaSet = []
for i in range(len(routine)):  # i starts from 0, routine has 705 elements
    while True:
        if i == len(routine) - 1:
            break
        else:
            i_point = routine[i][0].transform.location
            ii_point = routine[i + 1][0].transform.location

            d_i = get_player_car_distance(i_point.x, i_point.y)
            d_ii = get_player_car_distance(ii_point.x, ii_point.y)

            if d_ii >= d_i:
                TempList.append(d_i)
            elif d_ii < d_i:
                if TempList:
                    DeltaSet.append(min(TempList))
                    # f.write(str(i) + " ")
                    f.write(str(min(TempList)) + " ")
                    f.write(str(time.time() - start_time) + " ")
                    f.write(str(get_player_car_yaw()) + " ")
                    f.write(str(get_player_car_speed()) + "\n")
                    print("---------------------------------------")
                    print("Road Point" + str(i) + ":")
                    print("Relative Distance: " + str(min(TempList)))
                    print("Speed: " + str(get_player_car_speed()) + "km/h")
                    print("Time now is:" + str(time.time() - start_time))
                    TempList.clear()
                    break
                else:
                    f.write("Escaped Road Point" + str(i) + "\n")
                    print("---------------------------------------")
                    print("Escaped Point" + str(i))
                    print("d_i: " + str(d_i))
                    print("d_ii: " + str(d_ii))
                    print("d_ii - d_i: " + str(d_ii - d_i))
                    print("Waypoint: " + str(routine[i][0].transform.location))
                    # i = i + 10
                    break
f.close()
