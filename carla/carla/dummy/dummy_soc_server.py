import socket

# Create a TCP/IP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Connect the socket to the port where the server is listening
server_address = ('192.168.1.15', 12345)
client_address = ('192.168.1.15', 12346)

client_socket.bind(server_address)

try:
    # send data to server
    i = 0
    while True : 
        # Receive response
        data = client_socket.recvfrom(1024)
        print('Received', repr(data))
        print(i)

        # data = str(data_to_send[i % len(data_to_send)]).encode()
        data = "Hello, World!".encode()
        client_socket.sendto(data, client_address)
        i += 1



finally:
    # Clean up the connection
    client_socket.close()