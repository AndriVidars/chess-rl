import chess
from RL.game_environment.agent import Agent

class Game:
    def __init__(self, board: chess.Board, agent_white: Agent, agent_black: Agent):
        self.board = board
        self.agent_white = agent_white
        self.agent_black = agent_black
    
    def make_turn(self):
        if self.board.turn == chess.WHITE:
            move = self.agent_white.sample_move()
        else:
            move = self.agent_black.sample_move()        
        self.board.push(move)

    
    def get_result(self):
        assert self.board.is_game_over()
        outcome = self.board.outcome()
        if outcome.winner == chess.WHITE:
            return type(self.agent_white)
        elif outcome.winner == chess.BLACK:
            return type(self.agent_black)
        else:
            return None

    def play(self):
        while not self.board.is_game_over():
            self.make_turn()
            if self.board.is_game_over():
                return self.get_result()
    