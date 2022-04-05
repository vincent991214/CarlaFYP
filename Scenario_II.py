"""
https://github.com/carla-simulator/carla/issues/3890
"""
import math
import time
import carla
import numpy as np

f = open("Vincent_Samples/Delay_Free_Drive/100ms_Free_Drive_Town07_test.txt", "w")
f.write("x_coor y_coor Yaw Speed Time" + "\n")
start_time = time.time()

def get_player_car_attributes():
    for target in world.get_actors():
        if target.attributes.get('role_name') == 'hero':
            player = target
            velocity = player.get_velocity()
            transform = player.get_transform()
            # Get car Locations
            location_x = player.get_location().x
            location_y = player.get_location().y
            # Get car Speed
            vel_np = np.array([velocity.x, velocity.y, velocity.z])
            pitch = np.deg2rad(transform.rotation.pitch)
            # Get car Yaw
            yaw = np.deg2rad(transform.rotation.yaw)
            orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
            speed = np.dot(vel_np, orientation)
            # Jotting Down Measurements
            f.write(str(location_x) + " ")
            f.write(str(location_y) + " ")
            f.write(str(round(transform.rotation.yaw, 3)) + " " + str(round((3.6*speed), 2)) + " ")
            f.write(str(round(time.time()-start_time, 3)) + "\n")
            break
    return


client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.load_world("Town07")
blueprint_library = world.get_blueprint_library()

while True:
    time.sleep(0.1)
    get_player_car_attributes()

f.close()
