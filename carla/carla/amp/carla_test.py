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
    player.set_autopilot(True)
    actors.append(player)


    # spawn camera on last vehicle
    bp = blueprint_library.find("sensor.camera.rgb")
    camera = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=player)
    camera.listen(lambda image : image.save_to_disk("output/%.6d.png" % image.frame))


    amp = AMP(world, player)
    # for i in range(70) :
    i=0
    while True:
        i+=1 
        try:
            # Your code that operates on the actor
            batch = amp.get_amp_inputs()
            import time
            if len(batch) > 0 and i%1==0: 
                batch_idx = np.random.randint(0, len(batch))
                agents_vectors = batch[batch_idx][0].detach().numpy()
                objects_vectors = np.array(batch[batch_idx][1])
                lanes_vectors = np.array(batch[batch_idx][2])
                n_objects=0
                cd=0
                cnt=0
                for b in agents_vectors :
                    cd += b[:, -1].mean()
                    cnt+=1
                print("Average NO. Objects: ", cd / cnt)

                # # save vectors in disk
                # # batch_name = np.random.randint(0, 100000)
                # batch_name = time.time()
                # os.makedirs(f"amp_batches_2/{batch_name}")
                # np.save(f"amp_batches_2/{batch_name}/agents_vectors.npy", agents_vectors)
                # np.save(f"amp_batches_2/{batch_name}/objects_vectors.npy", objects_vectors)
                # np.save(f"amp_batches_2/{batch_name}/lanes_vectors.npy", lanes_vectors)
                        # if i%100 == 0 :  
                if len(batch) > 0 : 
                    outputs = amp.get_amp_outputs(batch)

                    for it, out in enumerate(outputs) : 
                        # Draw only for vehicles
                        object_vectors = amp.objects_vectors
                        if object_vectors[it][0][-1] not in [0, 1, 3] : 
                            continue

                        for i in range(len(out)-1) : 
                            sx, sy = out[i]
                            ex, ey = out[i+1]
                            # print(sx, sy, ex, ey)
                            start = carla.Location(x=float(sx), y=float(sy), z=1.0)
                            end = carla.Location(x=float(ex), y=float(ey), z=1.0)
                            world.debug.draw_line( start, end, thickness=0.1,  color=carla.Color(r=150, g=0, b=0), life_time=3.5)

        except RuntimeError as e:
            print(f"RuntimeError: {e}")

        # print(batch)
        world.tick()

finally : 
    cleanup(actors)