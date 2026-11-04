"""Dataset, worker isolation and real-engine integration for KQ benchmarks."""

import copy
import io
import json
import multiprocessing
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import chess
import chess.pgn

from benchmarks.kq import audit_record, board_for, load_dataset
from benchmarks.run_kq import (_agent_worker, main, parser_for_cli, receive,
                               run_game, stop_worker, TrialError)


class MateAgent:
    def choose_move(self, board):
        for move in board.legal_moves:
            private = board.copy()
            private.push(move)
            if private.is_checkmate():
                return move
        return next(iter(board.legal_moves))


class MutatingAgent:
    def choose_move(self, board):
        board.push(next(iter(board.legal_moves)))
        return next(iter(board.legal_moves))


class IllegalAgent:
    def choose_move(self, board):
        return chess.Move.null()


class SlowAgent:
    def __init__(self):
        time.sleep(2)

    def choose_move(self, board):
        return next(iter(board.legal_moves))


class KQBenchmarkTests(unittest.TestCase):
    def test_positions_are_legal_for_both_colors_and_include_requested_board(self):
        payload = load_dataset()
        self.assertEqual(len(payload['positions']), 4)
        requested = next(p for p in payload['positions'] if p['id'] == 'requested_d2_a2_e6')
        board = board_for(requested)
        self.assertEqual(board.piece_at(chess.D2), chess.Piece(chess.QUEEN, True))
        self.assertEqual(board.piece_at(chess.A2), chess.Piece(chess.KING, True))
        self.assertEqual(board.piece_at(chess.E6), chess.Piece(chess.KING, False))
        for p in payload['positions']:
            for color in chess.COLORS:
                board = board_for(p, color)
                self.assertTrue(board.is_valid())
                self.assertEqual(board.turn, color)
                self.assertEqual(len(board.piece_map()), 3)
                self.assertEqual(len(board.pieces(chess.QUEEN, color)), 1)

    def test_invalid_dataset_rejected(self):
        for fen in [chess.STARTING_FEN, '8/8/8/8/8/8/4k3/4K3 w - - 0 1',
                    '8/8/4k3/8/8/8/K2Q4/8 b - - 0 1']:
            payload = load_dataset()
            payload['positions'][0]['fen'] = fen
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'data.json'
                path.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_dataset(path)

    def test_replay_checks_mate_and_rejects_corrupted_san(self):
        p = next(p for p in load_dataset()['positions'] if p['category'] == 'control')
        board = board_for(p)
        initial = board.fen()
        move = MateAgent().choose_move(board)
        entry = {'ply': 1, 'actor': 'test', 'uci': move.uci(),
                 'san': board.san(move), 'fen_before': board.fen()}
        board.push(move)
        entry['fen_after'] = board.fen()
        record = {'agent': 'test', 'initial_fen': initial, 'final_fen': board.fen(),
                  'plies': 1, 'moves': [entry], 'result': '1-0', 'agent_result': 'win',
                  'termination': 'CHECKMATE'}
        self.assertTrue(audit_record(record)['passed'])
        bad = copy.deepcopy(record)
        bad['moves'][0]['san'] = 'Qa1'
        with self.assertRaises(ValueError):
            audit_record(bad)

    def test_unfinished_position_is_never_a_draw(self):
        board = board_for(load_dataset()['positions'][0])
        record = {'agent': 'test', 'initial_fen': board.fen(), 'final_fen': board.fen(),
                  'plies': 0, 'moves': [], 'result': '*', 'agent_result': 'unresolved',
                  'termination': 'max_plies'}
        self.assertTrue(audit_record(record)['passed'])
        record.update(result='1/2-1/2', agent_result='draw')
        with self.assertRaises(ValueError):
            audit_record(record)

    def worker(self, name):
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe()
        process = context.Process(target=_agent_worker,
                                  args=(child, __name__ + ':' + name, {}, 42, True, None))
        process.start()
        child.close()
        return process, parent

    def test_worker_reports_illegal_moves_and_input_mutation(self):
        board = board_for(load_dataset()['positions'][0])
        for name, status in [('IllegalAgent', 'illegal_move'), ('MutatingAgent', 'input_mutation')]:
            process, parent = self.worker(name)
            try:
                self.assertEqual(receive(parent, 10)['status'], 'ready')
                parent.send(('move', board))
                with self.assertRaises(TrialError) as error:
                    receive(parent, 10)
                self.assertEqual(error.exception.status, status)
            finally:
                stop_worker(process, parent, None)

    def test_worker_hard_startup_timeout_is_reported(self):
        process, parent = self.worker('SlowAgent')
        try:
            with self.assertRaises(TrialError) as error:
                receive(parent, .01)
            self.assertEqual(error.exception.status, 'timeout')
        finally:
            stop_worker(process, parent, None)

    def test_invalid_cli_budget_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['--max-plies', '0'])
        self.assertEqual(error.exception.code, 2)

    @unittest.skipUnless(sys.platform == 'win32', 'Bundled engine is a Windows executable')
    def test_real_stockfish_game_and_pgn_roundtrip(self):
        binary = Path(__file__).resolve().parents[1] / 'stockfish' / 'stockfish-windows-x86-64-universal.exe'
        if not binary.exists():
            self.skipTest('Bundled Stockfish unavailable')
        args = parser_for_cli().parse_args([])
        args.stockfish = str(binary)
        p = next(p for p in load_dataset()['positions'] if p['category'] == 'control')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            result = run_game('game', __name__ + ':MateAgent', p, True, 42, {}, args, output)
            self.assertIsNone(result['error'])
            self.assertEqual(result['agent_result'], 'win')
            self.assertEqual(result['termination'], 'CHECKMATE')
            self.assertEqual(result['stockfish']['options']['Skill Level'], 20)
            self.assertFalse(result['stockfish']['options']['UCI_LimitStrength'])
            game = chess.pgn.read_game(io.StringIO((output/'game.pgn').read_text(encoding='utf-8')))
            self.assertEqual(game.end().board().fen(), result['final_fen'])
            self.assertEqual(game.headers['Result'], '1-0')


if __name__ == '__main__':
    unittest.main()
