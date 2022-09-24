import os
import asyncio
import websockets
from ModelClass import ModelClass

model = ModelClass()
model.load_model()


async def recv_msg(websocket):
    while True:
        recv_text = await websocket.recv()
        if os.path.exists("runtime/"+recv_text):
            response_text = model.target_detect("runtime/"+recv_text)
        else:
            response_text = "没有该文件"
        await websocket.send(response_text)


async def main_logic(websocket):
    await recv_msg(websocket)


start_server = websockets.serve(main_logic, "127.0.0.1", "9898")
print("WS服务器已开启，可以开启客户端程序")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
