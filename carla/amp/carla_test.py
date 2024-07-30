import os
import sys
import glob 

sys.path.append('carla-0.9.9-py3.7-linux-x86_64.egg')
sys.path.append('/media/mahmoud/New Volume/faculty/level2/study/machine learning/Towards Enhanced Autonomous Vehicle/carla/')

import carla 
import numpy as np
from amp import AMP

def cleanup(actors):
    for actor in actors:
        try:
            actor.destroy()
            print(f"Destroyed actor {actor.id}")
        except Exception as e:
            print(f"Failed to destroy actor {actor.id}: {e}")
    print("Cleaned...")


try :
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(2.0)
    print("Connected to server")
    actors = []
    
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter("vehicle.*")[0]
    spawn_points = world.get_map().get_spawn_points()
    np.random.shuffle(spawn_points)

    # spawn player
    spawn_point = spawn_points[0]
    player = world.try_spawn_actor(bp, spawn_point)
    # player.set_autopilot(True)
    actors.append(player)


    # spawn camera on last vehicle
    bp = blueprint_library.find("sensor.camera.rgb")
    camera = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=player)
    camera.listen(lambda image : image.save_to_disk("output/%.6d.png" % image.frame))


    amp = AMP(world, player)
    for i in range(10) :
        try:
            # Your code that operates on the actor
            batch = amp.get_amp_inputs()
            n_objects = 0
            cnt=0
            if len(batch) > 0 : 
                for b in batch : 
                    n_objects += len(b[1])
                    cnt+=1
                print("Average NO. Objects: ", n_objects / cnt)
            print(50*"{}".format(i))
        except RuntimeError as e:
            print(f"RuntimeError: {e}")

        # print(batch)
        world.tick()

finally : 
    cleanup(actors)