import socket
import json
# Create a TCP/IP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Connect the socket to the port where the server is listening
server_address = ("192.168.1.15", 12345)
client_address = ("192.168.1.15", 12346)

client_socket.bind(client_address)
data_to_send = [
    {"gear" : "D", "steering" : 0.0, "throttle" :   90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 1.0, "throttle" :   90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 2.0, "throttle" :   90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 3.0, "throttle" :   90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 4.0, "throttle" :   90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 5.0, "throttle" :   90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 6.0, "throttle" :   90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 7.0, "throttle" :   90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 8.0, "throttle" :   90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 9.0, "throttle" :   90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 10.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 11.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 12.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 13.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 14.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 15.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 16.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 17.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 18.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 19.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 20.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 21.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 22.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 23.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 24.0, "throttle" :  90.0, "brake" : 0.0, }, 
    {"gear" : "D", "steering" : 25.0, "throttle" :  90.0, "brake" : 0.0, },
]
try:
    # send data to server
    i = 0
    while True : 
        data = json.dumps(data_to_send[i % len(data_to_send)]).encode('utf-8')
        client_socket.sendto(data, server_address)
        i += 1
        print(i)
        # Receive response
        data = client_socket.recv(1024)
        print("Received", repr(data))


finally:
    # Clean up the connection
    client_socket.close()