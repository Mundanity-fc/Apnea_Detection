import asyncio
import websockets
from ModelClass import ModelClass

model = ModelClass()
model.load_model()

async def check_permission(websocket):
    while True:
        recv_str = await websocket.recv()
        cred_dict = recv_str.split(":")
        if cred_dict[0] == "apnea" and cred_dict[1] == "apnea":
            response_str = "Authorization Corrected\r\n"
            await websocket.send(response_str)
            return True
        else:
            response_str = "Authorization Failed\r\n"
            await websocket.send(response_str)


async def recv_msg(websocket):
    while True:
        recv_text = await websocket.recv()
        response_text = "Received"
        await websocket.send(response_text)


async def main_logic(websocket, path):
    await check_permission(websocket)
    await recv_msg(websocket)


start_server = websockets.serve(main_logic, '127.0.0.1', '9898')

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
