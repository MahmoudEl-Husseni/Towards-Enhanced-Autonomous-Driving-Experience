import socket
import json

class STM(object):
        def __init__(self):
            # this is your IP address
            self.server_IP = ('192.168.1.101', 12345)
            # this is the IP address of the STM 
            self.STM_IP = ('192.168.1.141', 12346)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(self.server_IP)
            self.sock.setblocking(False)

        def send(self, autoP, speed, collision, aGear, BL, BR, handBrake):
            data = {
                "speed": speed,
                "alert": collision,
                "autoPilot": int(autoP),
                "autoGear": aGear,
                "leftBlink": BL,
                "rightBlink": BR,
                "warning": 0,
                "handBrake": int(handBrake)
            }
            try :
                json_data = json.dumps(data)
                self.sock.sendto(json_data.encode('utf-8'), self.STM_IP)
            except Exception as e:
                print("Error:", e)
        
        def receive(self):
            # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                # receive all data in buffer
                data, server = self.sock.recvfrom(4096)
                data = json.loads(data.decode())
                print(data)
                if data is None: 
                    return {}
                else :  
                    return data
            except Exception as e:
                print("Error:", e)

        def destroy (self) : 
            self.sock.close()
