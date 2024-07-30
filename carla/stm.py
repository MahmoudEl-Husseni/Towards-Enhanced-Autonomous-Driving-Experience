import socket
import json

class STM(object):
        def __init__(self):
            # this is your IP address
            self.server_IP = ('192.168.1.15', 12345)
            # this is the IP address of the STM 
            self.STM_IP = ('192.168.1.15', 12346)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(self.server_IP)

        def send(self, autoPilot, speed, gear, throttle, steer, brake, leftBlinker, reftBlinker ,warning ,alert):
            data = {
                "autoPilot": autoPilot,
                "speed": speed,
                "gear": gear,
                "throttle": throttle,
                "steer": steer,
                "brake": brake,
                "leftBlinker": leftBlinker,
                "reftBlinker": reftBlinker,
                "warning": warning,
                "alert": alert,
            }
            try :
                json_data = json.dumps(data)
                self.sock.sendto(json_data.encode('utf-8'), self.STM_IP)
            except Exception as e:
                print("Error:", e)
        
        def receive(self):
            # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                data, server = self.sock.recvfrom(4096)
                data = json.loads(data.decode())
                print(data)
                return  data
            except Exception as e:
                print("Error:", e)

        def destroy (self) : 
            self.sock.close()
