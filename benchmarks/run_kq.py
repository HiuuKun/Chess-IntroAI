"""Play any noninteractive agent's king and queen against Stockfish's lone king."""

import argparse
import concurrent.futures
import hashlib
import inspect
import json
import math
import multiprocessing
import os
import random
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from .kq import DEFAULT_DATASET, audit_record, board_for, load_dataset
from .run_999 import agent_options, get_factory


class TrialError(Exception):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


def write_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def _agent_worker(connection, name, options, seed, color, time_limit):
    agent = None
    try:
        random.seed(seed)
        factory = get_factory(name)
        bound = dict(options)
        if 'seed' in inspect.signature(factory).parameters:
            bound['seed'] = seed
        agent = factory(**bound)
        if getattr(agent, 'interactive', False):
            raise ValueError('Interactive agents cannot run headlessly')
        getattr(agent, "on_game_start", lambda color: None)(color)
        source = inspect.getsourcefile(type(agent))
        config = {k: getattr(agent, k) for k in (
            'simulations', 'rollout_depth', 'exploration', 'seed',
            'progressive_expansion', 'exchange_safety', 'rollout_randomness',
            'widening_base', 'widening_exponent') if hasattr(agent, k)}
        connection.send({'status': 'ready', 'config': config,
                         'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest()
                         if source else None})
        while True:
            action, value = connection.recv()
            if action == 'end':
                if value is not None:
                    getattr(agent, "on_game_end", lambda result: None)(value)
                break
            board = value
            before = board.fen(), tuple(board.move_stack)
            started = time.perf_counter()
            move = (agent.choose_move(board) if time_limit is None
                    else agent.choose_move(board, time_limit=time_limit))
            if before != (board.fen(), tuple(board.move_stack)):
                raise TrialError('input_mutation', 'Agent modified board/history')
            if not isinstance(move, chess.Move) or move not in board.legal_moves:
                raise TrialError('illegal_move', f'Illegal move: {move}')
            stats = getattr(agent, 'last_stats', {})
            stats = {str(k): v for k, v in stats.items()
                     if v is None or type(v) in (int, float, str, bool)} if isinstance(stats, dict) else {}
            connection.send({'status': 'move', 'move': move, 'stats': stats,
                             'elapsed_s': time.perf_counter() - started})
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send({'status': getattr(exc, 'status', 'agent_error'),
                             'error': f'{type(exc).__name__}: {exc}'})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if agent is not None:
            try:
                getattr(agent, "close", lambda: None)()
            except Exception:
                pass
        connection.close()


def receive(connection, timeout):
    if timeout <= 0 or not connection.poll(timeout):
        raise TrialError('timeout', 'Agent exceeded the hard response/game deadline')
    try:
        reply = connection.recv()
    except EOFError as exc:
        raise TrialError('agent_error', 'Agent worker exited without a result') from exc
    if reply['status'] not in ('ready', 'move'):
        raise TrialError(reply['status'], reply.get('error', 'Worker failure'))
    return reply


def stop_worker(process, connection, result):
    try:
        if process.pid is not None and process.is_alive():
            try:
                connection.send(('end', result))
            except (OSError, EOFError):
                pass
            process.join(.3)
            if process.is_alive():
                process.terminate()
                process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join()
    finally:
        connection.close()
        if process.pid is not None:
            process.close()


def run_game(game_id, name, position, color, seed, options, args, output, cancelled=None):
    board = board_for(position, color)
    record = {'id': game_id, 'agent': name, 'position_id': position['id'],
              'category': position['category'], 'color': 'white' if color else 'black',
              'seed': seed, 'agent_options': options, 'initial_fen': board.fen(),
              'moves': [], 'error': None, 'status': 'finished'}
    started = time.perf_counter()
    deadline = started + args.game_timeout
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=_agent_worker,
                              args=(child, name, options, seed, color, args.time_limit))
    engine = None
    try:
        process.start()
        child.close()
        reply = receive(parent, min(args.timeout, deadline - time.perf_counter()))
        record.update(agent_config=reply['config'], agent_source_sha256=reply['source_sha256'])
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish, timeout=10)
        skill = engine.options['Skill Level']
        settings = {'Skill Level': args.stockfish_skill, 'UCI_LimitStrength': False,
                    'Threads': 1, 'Hash': 32}
        engine.configure(settings)
        record['stockfish'] = {'id': engine.id, 'options': settings,
                               'maximum_skill': skill.max, 'move_time_s': args.stockfish_time}
        with (output / f'{game_id}.moves.jsonl').open('w', encoding='utf-8') as stream:
            while board.outcome(claim_draw=True) is None and len(record['moves']) < args.max_plies:
                if cancelled is not None and cancelled.is_set():
                    raise TrialError('cancelled', 'Benchmark interrupted')
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TrialError('timeout', 'Game exceeded its hard wall limit')
                entry = {'ply': len(record['moves']) + 1, 'fen_before': board.fen()}
                if board.turn == color:
                    parent.send(('move', board.copy(stack=True)))
                    reply = receive(parent, min(args.timeout, remaining))
                    move = reply['move']
                    entry.update(actor=name, elapsed_s=reply['elapsed_s'], stats=reply['stats'])
                else:
                    t0 = time.perf_counter()
                    response = engine.play(board, chess.engine.Limit(time=min(args.stockfish_time, remaining)))
                    move = response.move
                    entry.update(actor='Stockfish', elapsed_s=time.perf_counter() - t0)
                if not isinstance(move, chess.Move) or move not in board.legal_moves:
                    raise TrialError('illegal_move', f'Invalid move from {entry["actor"]}: {move}')
                entry.update(uci=move.uci(), san=board.san(move))
                board.push(move)
                entry['fen_after'] = board.fen()
                record['moves'].append(entry)
                stream.write(json.dumps(entry) + '\n')
                stream.flush()
        if board.outcome(claim_draw=True) is None:
            record['status'] = 'max_plies'
    except TrialError as exc:
        record.update(status=exc.status, error=str(exc))
    except Exception as exc:
        record.update(status='runner_error', error=f'{type(exc).__name__}: {exc}')
    finally:
        child.close()
        outcome = board.outcome(claim_draw=True)
        stop_worker(process, parent, outcome.result() if outcome else None)
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                engine.close()
        record.update(plies=len(record['moves']), final_fen=board.fen(),
                      elapsed_s=time.perf_counter() - started,
                      result=outcome.result() if outcome else '*',
                      termination=outcome.termination.name if outcome else record['status'],
                      agent_result=('draw' if outcome.winner is None else 'win' if outcome.winner == color else 'loss')
                      if outcome else 'unresolved')
        record['audit'] = audit_record(record)
        game = chess.pgn.Game.from_board(board)
        game.headers.update(Event='King and queen vs king', Site='Local benchmark',
                            Date=datetime.now(timezone.utc).strftime('%Y.%m.%d'), Round=game_id,
                            White=name if color else f'Stockfish Skill {args.stockfish_skill}',
                            Black=f'Stockfish Skill {args.stockfish_skill}' if color else name,
                            Result=record['result'], Termination=record['termination'])
        (output / f'{game_id}.pgn').write_text(str(game) + '\n', encoding='utf-8')
        write_json(output / f'{game_id}.json', record)
    return record


def summarize(records):
    return {name: {category: dict(Counter(r['agent_result'] for r in records
                                        if r['agent'] == name and (category == 'all' or r['category'] == category)))
                   for category in ['all', 'general', 'control']}
            for name in dict.fromkeys(r['agent'] for r in records)}


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', nargs='+', default=['MCTS', 'mcts_improve'])
    parser.add_argument('--simulations', type=int, default=200)
    parser.add_argument('--rollout-depth', type=int)
    parser.add_argument('--exploration', type=float)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--colors', nargs='+', choices=['white', 'black'], default=['white', 'black'])
    parser.add_argument('--position-ids', nargs='+')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--workers', type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument('--timeout', type=float, default=60, help='Hard seconds per agent response, including startup')
    parser.add_argument('--game-timeout', type=float, default=600)
    parser.add_argument('--time-limit', type=float, help='Optional cooperative seconds per agent move')
    parser.add_argument('--max-plies', type=int, default=200)
    parser.add_argument('--stockfish', type=str)
    parser.add_argument('--stockfish-skill', type=int, default=20)
    parser.add_argument('--stockfish-time', type=float, default=.2)
    extras = parser.add_mutually_exclusive_group()
    extras.add_argument('--agent-options')
    extras.add_argument('--options-file', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--list-agents', action='store_true')
    return parser


def main(argv=None):
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    if args.list_agents:
        from agents import AGENT_FACTORIES
        print('\n'.join(AGENT_FACTORIES))
        print('Custom agents: module:Class')
        return 0
    if min(args.simulations, args.workers, args.max_plies) <= 0:
        parser.error('simulations, workers and max-plies must be positive')
    if not 0 <= args.stockfish_skill <= 20:
        parser.error('Stockfish skill must be 0..20')
    for field in ['timeout', 'game_timeout', 'stockfish_time']:
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'{field} must be finite and positive')
    if args.time_limit is not None and (not math.isfinite(args.time_limit) or args.time_limit < 0):
        parser.error('time-limit must be finite and nonnegative')
    if any(len(set(values)) != len(values) for values in [args.agent, args.colors, args.seeds]):
        parser.error('agents, colors and seeds must not repeat')
    try:
        payload = load_dataset(args.dataset)
        supplied = json.loads(args.options_file.read_text(encoding='utf-8-sig')
                              if args.options_file else args.agent_options or '{}')
        if (not isinstance(supplied, dict) or set(supplied) - set(args.agent)
                or any(not isinstance(v, dict) for v in supplied.values())):
            raise ValueError('Options must map selected agent names to constructor options')
        options = {name: agent_options(name, args, supplied.get(name, {})) for name in args.agent}
        positions = payload['positions']
        if args.position_ids:
            if set(args.position_ids) - {p['id'] for p in positions}:
                raise ValueError('Unknown position ID')
            positions = [p for p in positions if p['id'] in args.position_ids]
    except (OSError, ValueError, TypeError, ImportError, AttributeError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print(json.dumps(payload, indent=2))
        return 0
    try:
        from agents.stockfish_agent import StockfishAgent
        if args.stockfish and not Path(args.stockfish).is_file():
            raise ValueError('Explicit Stockfish path does not exist')
        args.stockfish = StockfishAgent(path=args.stockfish)._find_binary()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or Path('benchmark_results') / ('kq-' + stamp)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(f'Cannot create a new output directory: {exc}')
    metadata = {'complete': False, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'dataset': payload, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'agent_options': options, 'scoring': 'Completed chess outcomes with claim_draw=True; limits and failures are unresolved, not draws or losses.'}
    records = []
    jobs = [(f'g{i:04d}', name, position, color == 'white', seed)
            for i, (name, position, color, seed) in enumerate(
                ((a, p, c, s) for a in args.agent for p in positions for c in args.colors for s in args.seeds), 1)]
    metadata['expected_games'] = len(jobs)
    cancelled = threading.Event()
    def checkpoint():
        write_json(output / 'summary.json', {'metadata': metadata, 'summary': summarize(records),
                   'status_counts': dict(Counter(r['status'] for r in records)), 'games': records})
    checkpoint()
    print(f'Output: {output.resolve()}', flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_game, gid, name, p, c, seed, options[name], args, output, cancelled)
                       for gid, name, p, c, seed in jobs]
            try:
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    records.append(record)
                    checkpoint()
                    print(f"[{len(records)}/{len(jobs)}] {record['agent']} {record['position_id']} "
                          f"{record['color']}: {record['agent_result']} ({record['termination']})", flush=True)
            except KeyboardInterrupt:
                cancelled.set()
                for future in futures:
                    future.cancel()
                raise
    except KeyboardInterrupt:
        metadata['interrupted'] = True
        checkpoint()
        return 130
    metadata.update(complete=True, finished_at_utc=datetime.now(timezone.utc).isoformat())
    checkpoint()
    lines = ['# King and queen versus king', '',
             'Outcomes are completed games against Stockfish. General positions and mate-in-one controls are separated.', '',
             '| Agent | Position | Color | Result | Reason | Plies | PGN |',
             '|---|---|---|---|---|---:|---|']
    for r in sorted(records, key=lambda r: r['id']):
        lines.append(f"| {r['agent']} | {r['position_id']} | {r['color']} | {r['agent_result']} | "
                     f"{r['termination']} | {r['plies']} | [Game]({r['id']}.pgn) |")
    lines += ['', 'Limits and errors remain unresolved. See summary.json for settings, status counts and replay audits.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summarize(records), indent=2))
    return int(any(r['status'] not in ('finished', 'max_plies') for r in records))


if __name__ == '__main__':
    raise SystemExit(main())
