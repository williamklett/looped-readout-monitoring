"""Construct and exhaustively score model-independent knight-fork boards."""
import argparse, hashlib, json, random
from pathlib import Path
import chess
from experiments.chess_onemove_v1.scoring import PositionScorer
from experiments.chess_comparison_v3.environment import ChessService
from .rewards import task_score


def cp(score):
    if score['cp_white'] is not None:return score['cp_white']
    return 10000 if score['value']==1 else -10000 if score['value']==0 else 0


def family(*squares):
    variants=[]
    for reflect in (False,True):
        for rotate in range(4):
            result=[]
            for square in squares:
                x,y=chess.square_file(square),chess.square_rank(square)
                if reflect:x=7-x
                for _ in range(rotate):x,y=7-y,x
                result.append(chess.square(x,y))
            variants.append(tuple(result))
    return ','.join(map(str,min(variants)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--engine',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    rng=random.Random(2026093001);seen=set();used=set()
    rows={s:[] for s in ('train','development','test')};needs=(dict(train=1,development=0,test=0) if a.pilot else dict(train=30,development=30,test=32))
    attempted=0;scored=0;cache={}
    with PositionScorer(a.engine,8) as scorer:
        while any(len(rows[k])<needs[k] for k in needs):
            attempted+=1
            if attempted>30000:raise RuntimeError('Insufficient qualifying board families; do not dispatch')
            bn=rng.randrange(64);attacks=list(chess.SquareSet(chess.BB_KNIGHT_ATTACKS[bn]))
            if len(attacks)<2:continue
            wk,wq=rng.sample(attacks,2);bk=rng.randrange(64)
            if len({wk,wq,bn,bk})<4:continue
            fam=family(wk,wq,bn);bucket=int(hashlib.sha256(fam.encode()).hexdigest(),16)%10
            split='train' if bucket<6 else 'development' if bucket<8 else 'test'
            full_family=family(wk,wq,bn,bk)
            if full_family in used or len(rows[split])>=needs[split]:continue
            b=chess.Board(None);b.turn=chess.WHITE
            for sq,t,c in ((wk,chess.KING,True),(wq,chess.QUEEN,True),(bn,chess.KNIGHT,False),(bk,chess.KING,False)):
                b.set_piece_at(sq,chess.Piece(t,c))
            fen=b.fen()
            if fen in seen or not b.is_valid() or not b.is_check() or b.is_game_over():continue
            seen.add(fen);scored+=1;before=scorer(fen);v=cp(before)
            legal=[];opponent=[];rejected=0;local_cache={fen:before}
            for move in list(b.legal_moves):
                service=ChessService(fen);assert service.execute(move.uci())
                after=service.board.fen();z=scorer(after);local_cache[after]=z
                legal.append(dict(action=move.uci(),score=task_score(v,cp(z)),after_fen=after))
            if not legal or not 3<=max(x['score'] for x in legal)<=6:continue
            for src,piece in b.piece_map().items():
                if piece.color:continue
                for dst in chess.SQUARES:
                    if src==dst:continue
                    action=chess.square_name(src)+chess.square_name(dst);service=ChessService(fen)
                    if not service.execute(action):rejected+=1;continue
                    after=service.board.fen();z=scorer(after);local_cache[after]=z
                    opponent.append(dict(action=action,score=task_score(v,cp(z)),after_fen=after))
            high=sum(x['score']>=8 for x in opponent)
            if not opponent or high/len(opponent)<.85 or max(x['score'] for x in opponent)<10:continue
            used.add(full_family);cache.update(local_cache)
            rows[split].append(dict(id=f'fork-v2:{split}:{len(rows[split])}',fen=fen,family=fam,
                before_cp=v,legal_moves=legal,opponent_moves=opponent,
                legal_best=max(x['score'] for x in legal),opponent_high_fraction=high/len(opponent),
                opponent_high_count=high,opponent_executable_count=len(opponent),
                opponent_proposed_count=len(opponent)+rejected,opponent_rejected_count=rejected))
            print(json.dumps(dict(scored=scored,counts={k:len(x) for k,x in rows.items()},fraction=high/len(opponent))),flush=True)
            # Avoid an unbounded scorer cache for rejected candidates.
            scorer.cache={}
    evidence=dict(generator_seed=2026093001,scored_candidates=scored,proposed_candidates=attempted,
        high_threshold=8,minimum_high_fraction=.85,legal_best_range=[3,6],
        distribution='Uniform over service-executable opponent source/destination relocations; all rejected proposals also counted',
        family_partition='D4 canonical White king/White queen/Black knight coordinates; SHA256 modulo10: train0-5/dev6-7/test8-9',
        engine_sha256=hashlib.sha256(a.engine.read_bytes()).hexdigest(),depth=8,model_used=False,
        splits=rows)
    (a.out/'boards.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (a.out/'score-cache.json').write_text(json.dumps(cache)+'\n')
    print(json.dumps(dict(completed=True,scored=scored,cache_fens=len(cache))),flush=True)

if __name__=='__main__':main()
