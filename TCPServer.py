import socket
import os

# 总体参数
IP = "127.0.0.1"
PORT = 4455
ADDR = (IP, PORT)
FORMAT = "utf-8"
SIZE = 512000
workdir = os.path.dirname(os.path.abspath(__file__)) + '/'


def main():
    print("[开始] TCP服务器正在启动")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    print("[监听] TCP服务器正在监听")
    while True:
        conn, addr = server.accept()
        print(f"[新的连接] {addr} 已连接至本服务器")
        filename = conn.recv(SIZE).decode(FORMAT)
        print(f"[获取] 获取到指定文件名")
        file = open(workdir + 'runtime/'+filename, "wb")
        conn.send("文件开始接收".encode(FORMAT))
        data = conn.recv(SIZE)
        print(f"[获取] 获取到指定文件内容")
        file.write(data)
        conn.send("文件数据已保存".encode(FORMAT))
        file.close()
        conn.close()
        print(f"[断开连接] {addr} 已断开与本服务器的连接")


if __name__ == "__main__":
    main()
