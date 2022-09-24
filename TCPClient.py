import socket

# 总体参数
IP = input("请键入目标IP：")
PORT = 4455
ADDR = (IP, PORT)
FORMAT = "utf-8"
SIZE = 512000


def main():
    """ Staring a TCP socket. """
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)
    file = open("runtime/TestDataA.mat", "rb")
    data = file.read()
    client.send("abc.mat".encode(FORMAT))
    msg = client.recv(SIZE).decode(FORMAT)
    print(f"[SERVER]: {msg}")
    client.send(data)
    msg = client.recv(SIZE).decode(FORMAT)
    print(f"[SERVER]: {msg}")
    file.close()
    client.close()


if __name__ == "__main__":
    main()
