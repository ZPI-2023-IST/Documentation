Build your own API module
============================

===========================
How to build the module
===========================

You can build API module in any way you like. The only requirement is that the module needs to be able to connect to RL module

======================================================
How to test if the module was implemented correctly
======================================================

| API module should be able to load Game and Translator module of your choice
| It should be able to connect to RL module
| `Postman is a good tool to verify if connection works <https://www.postman.com/>`__.

======================================================
Game module example (Freecell)
======================================================

""""""""""""""""""""""""""""""""""""""
Module structure
""""""""""""""""""""""""""""""""""""""

Among many files, our core functionality was split onto following files:
    * ``main.py`` - contains API module implementation
    * ``runner.py`` - loads Game and Translator modules

""""""""""""""""""""""""""""""""""""""
main.py
""""""""""""""""""""""""""""""""""""""

For the full implementation look at `our repository <https://github.com/ZPI-2023-IST/API>`__.
Let's take a look at the API module implementation::

    import json

    import socketio
    from aiohttp import web

    from runner import Runner

    sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")
    app = web.Application()
    runner = Runner()
    sio.attach(app)


    def parse_board(board):
        new_board =   {"Board": [], "FreeCells": [], "Stack": []}
        for stack in board[0]:
            new_board["Board"].append([])
            for card in stack:
                card = card.__repr__() if card is not None else None
                new_board["Board"][-1].append(card)
        for card in board[1]:
            card = card.__repr__() if card is not None else None
            new_board["FreeCells"].append(card)
        for card in board[2]:
            card = card.__repr__() if card is not None else None
            new_board["Stack"].append(card)
        return new_board


    @sio.event
    async def make_move(sid, data):
        data = json.loads(data)
        move_ml = data.get("move")

        print(data)
        print(f"sender: {sid} tells translator to make move:")
        print(f"move_ml: {move_ml}")
        
        if move_ml is None:
            runner.reset()
        else:
            runner.translator.make_move(move_ml)
        
        state = runner.translator.get_state()
        reward = runner.translator.get_reward()
        moves = runner.translator.get_moves()
        board = runner.translator.get_board()
        board_raw = runner.game.get_board()
            
        print(f"moves: {moves}")

        response_data = {
            "moves_vector": moves,
            "game_board": board,
            "reward": reward,
            "state": state.name,
            "board_raw": parse_board(board_raw)
        }

        await sio.emit("get_response", json.dumps(response_data), room=sid)



    if __name__ == '__main__':
        web.run_app(app, host="0.0.0.0", port=5002)

======================================================
Game module example (2048)
======================================================

""""""""""""""""""""""""""""""""""""""
Module structure
""""""""""""""""""""""""""""""""""""""

Among many files, our core functionality was split onto following files:
    * ``main.py`` - contains API module implementation
    * ``runner.py`` - loads Game and Translator modules

""""""""""""""""""""""""""""""""""""""
main.py
""""""""""""""""""""""""""""""""""""""

For the full implementation look at `our repository <https://github.com/ZPI-2023-IST/API_2048>`__.
Let's take a look at the API module implementation::

    import json

    import socketio
    from aiohttp import web

    from runner import Runner

    sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")
    app = web.Application()
    runner = Runner()
    sio.attach(app)


    def parse_board(board):
        new_board = [
            [x.value if x.value is not None else 0 for x in row]
            for row in board
        ]

        return new_board


    @sio.event
    async def make_move(sid, data):
        data = json.loads(data)
        move_ml = data.get("move")

        if move_ml is None:
            runner.reset()
        else:
            runner.translator.make_move(move_ml)

        state = runner.translator.get_state()
        reward = runner.translator.get_reward()
        moves = runner.translator.get_moves()
        board = runner.translator.get_board()
        board_raw = runner.game.get_board()

        response_data = {
            "moves_vector": moves,
            "game_board": board,
            "reward": reward,
            "state": state.name,
            "board_raw": parse_board(board_raw)
        }

        print(response_data)

        await sio.emit("get_response", json.dumps(response_data), room=sid)


    if __name__ == '__main__':
        web.run_app(app, host="0.0.0.0", port=5002)
