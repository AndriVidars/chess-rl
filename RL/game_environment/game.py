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

    
    def get_winner(self, claim_draw=True) -> Agent | None:
        assert self.board.is_game_over(claim_draw=claim_draw)
        outcome = self.board.outcome(claim_draw=claim_draw)
        if outcome.winner == chess.WHITE:
            return self.agent_white
        elif outcome.winner == chess.BLACK:
            return self.agent_black
        else:
            return None
    
    # NOTE: Not used currently, will ne# TODO use game.play() instead of this loop? or will need to change game.play() to support batching/move caching/parallelism in train and evals?
    def play(self):
        while not self.board.is_game_over(claim_draw=True):
            self.make_turn()
            if self.board.is_game_over(claim_draw=True):
                return self.get_winner()
    