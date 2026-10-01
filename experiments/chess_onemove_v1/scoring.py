"""Deterministic external position scoring; never exposed to the policy."""
import math
import chess
import chess.engine


def bounded_value(cp):
    # A bounded utility, not a calibrated probability of winning.
    return (1.0 + math.tanh(float(cp) / 600.0)) / 2.0


class PositionScorer:
    def __init__(self, executable, depth=8):
        self.engine = chess.engine.SimpleEngine.popen_uci(str(executable), timeout=60.)
        self.engine.configure({'Threads': 1, 'Hash': 32})
        self.depth = depth
        self.cache = {}

    def __enter__(self): return self
    def __exit__(self, *_): self.engine.quit()

    def __call__(self, fen):
        if fen not in self.cache:
            board = chess.Board(fen)
            assert board.is_valid(), 'Scorer requires a valid board'
            outcome = board.outcome(claim_draw=True)
            if outcome:
                value = .5 if outcome.winner is None else float(outcome.winner == chess.WHITE)
                result = dict(value=value, cp_white=None, mate_white=None,
                              terminal=outcome.result(), best_legal_move=None)
            else:
                # Clear transpositions and start a new game for every independent
                # FEN. Single-thread fixed-depth scores cannot depend on run order.
                self.engine.configure({'Clear Hash': None})
                info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth), game=object())
                score = info['score'].white()
                cp = score.score(mate_score=10000)
                result = dict(value=bounded_value(cp), cp_white=cp,
                    mate_white=score.mate(), terminal=None,
                    best_legal_move=info['pv'][0].uci(), search_depth=info['depth'])
            assert 0 <= result['value'] <= 1
            self.cache[fen] = result
        return dict(self.cache[fen])
