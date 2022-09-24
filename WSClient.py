import asyncio
import websockets

ip = input("请键入目标主机IP：")


async def send_msg(websocket):
    while True:
        _text = input("键入目标文件名或键入exit退出: ")
        if _text == "exit":
            print(f"您已退出")
            await websocket.close(reason="user exit")
            return False
        await websocket.send(_text)
        recv_text = await websocket.recv()
        print(f"{recv_text}")


async def main_logic():
    async with websockets.connect("ws://" + ip + ":9898") as websocket:
        await send_msg(websocket)


asyncio.get_event_loop().run_until_complete(main_logic())
