"""Material-safety regressions; no external engine or tablebase required."""

import random
import unittest
from unittest.mock import patch

import chess

from agents.mcts_improve import MCTSImproveAgent, _Node, _SearchTimeout


class ExchangeSafetyTests(unittest.TestCase):
    def setUp(self):
        self.agent = MCTSImproveAgent(seed=42)

    def test_hanging_every_piece_type_for_both_colors(self):
        fixtures = (
            ("8/8/8/8/8/1k4K1/8/Q7 w - - 90 46", "a1a2", 900),
            ("8/8/8/8/8/1k4K1/8/R7 w - - 0 1", "a1a2", 500),
            ("8/8/8/8/8/1k4K1/8/B7 w - - 0 1", "a1b2", 330),
            ("8/8/8/8/8/1kN5/8/6K1 w - - 0 1", "c3a2", 320),
            ("8/8/8/8/1k6/8/P7/6K1 w - - 0 1", "a2a3", 100),
        )
        for fen, uci, value in fixtures:
            for mirror in (False, True):
                board = chess.Board(fen)
                move = chess.Move.from_uci(uci)
                if mirror:
                    board = board.mirror()
                    move = chess.Move(chess.square_mirror(move.from_square),
                                      chess.square_mirror(move.to_square))
                with self.subTest(piece=value, mirror=mirror):
                    self.assertTrue(board.is_valid())
                    self.assertIn(move, board.legal_moves)
                    before = board.fen(), tuple(board.move_stack)
                    self.assertEqual(self.agent._move_safety(board, move), -value)
                    self.assertEqual((board.fen(), tuple(board.move_stack)), before)

    def test_exchange_accounts_for_recaptures_and_profitable_sacrifices(self):
        fixtures = (
            # Qxd5 Rxd5: pawn gained, queen lost.
            ("3r2k1/8/8/3p4/8/8/8/3Q2K1 w - - 0 1", "d1d5", -800),
            # Bxc6 bxc6: bishop traded for a rook, a good exchange.
            ("6k1/1p6/2r5/8/8/8/6B1/6K1 w - - 0 1", "g2c6", 170),
            # Rxd5 Rxd5 Bxd5: the rook is defended; Black can decline.
            ("3r2k1/8/8/3p4/8/8/6B1/3R2K1 w - - 0 1", "d1d5", 100),
            # A second rook is revealed behind the first recapturing rook.
            ("3r2k1/3r4/8/3p4/8/8/6B1/3R2K1 w - - 0 1", "d1d5", -230),
        )
        for fen, uci, expected in fixtures:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            with self.subTest(uci=uci, expected=expected):
                self.assertTrue(board.is_valid())
                self.assertIn(move, board.legal_moves)
                before = board.fen(), tuple(board.move_stack)
                self.assertEqual(self.agent._see(board, move), expected)
                self.assertEqual(self.agent._move_safety(board, move), expected)
                self.assertEqual((board.fen(), tuple(board.move_stack)), before)

    def test_a_quiet_move_can_leave_another_piece_hanging(self):
        board = chess.Board("r6k/8/8/8/8/8/Q7/6K1 w - - 0 1")
        self.assertEqual(self.agent._move_safety(board, chess.Move.from_uci("g1h1")), -900)
        self.assertEqual(self.agent._move_safety(board, chess.Move.from_uci("a2b2")), 0)

    def test_illegal_pinned_attacker_does_not_enter_exchange(self):
        board = chess.Board("4k3/3Nr3/8/8/8/7B/8/4R1K1 b - - 0 1")
        self.assertTrue(board.is_valid())
        self.assertNotIn(chess.Move.from_uci("e7d7"), board.legal_moves)
        self.assertEqual(self.agent._exchange_reply(board, chess.D7, None), 0)

    def test_defended_queen_cannot_be_captured_by_king(self):
        board = chess.Board("8/8/8/8/8/1k6/8/QK6 w - - 0 1")
        move = chess.Move.from_uci("a1a2")
        self.assertEqual(self.agent._move_safety(board, move), 0)
        board.push(move)
        self.assertNotIn(chess.Move.from_uci("b3a2"), board.legal_moves)

    def test_en_passant_and_capture_promotion_gains(self):
        fixtures = (
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 100),
            ("r6k/1P6/8/8/8/8/8/6K1 w - - 0 1", "b7a8q", 1300),
            ("r6k/1P6/8/8/8/8/8/6K1 w - - 0 1", "b7a8n", 720),
            ("r6k/1P6/r7/8/8/8/8/6K1 w - - 0 1", "b7a8q", 400),
        )
        for fen, uci, expected in fixtures:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            with self.subTest(uci=uci):
                self.assertIn(move, board.legal_moves)
                self.assertEqual(self.agent._see(board, move), expected)

    def test_rollout_random_branch_also_avoids_hanging_pieces(self):
        board = chess.Board("8/8/8/8/8/1k4K1/8/Q7 w - - 90 46")
        moves = list(board.legal_moves)
        self.agent.rollout_randomness = 1.0
        rng = random.Random(0)
        for _ in range(50):
            selected = self.agent._rollout_move(board, moves, rng)
            self.assertGreaterEqual(self.agent._move_safety(board, selected), 0)

    def test_all_unsafe_rollout_moves_remain_playable(self):
        board = chess.Board("8/8/8/8/8/1k4K1/8/Q7 w - - 0 1")
        moves = [chess.Move.from_uci("a1a2"), chess.Move.from_uci("g3h3")]
        # Force an all-unsafe choice, as can happen when material loss is forced.
        with patch.object(self.agent, "_move_safety",
                          side_effect=lambda board, move, deadline: -900 if move == moves[0] else -100):
            self.assertEqual(self.agent._rollout_move(board, moves, random.Random(0)), moves[1])

    def test_final_tie_prefers_safety_but_proven_results_still_win(self):
        hanging = _Node(turn=False, move=chess.Move.from_uci("a1a2"),
                        visits=7, value=3.5, safety=-900)
        safe = _Node(turn=False, move=chess.Move.from_uci("a1c1"),
                     visits=7, value=3.5, safety=0)
        root = _Node(turn=True, children=[hanging, safe])
        self.assertIs(self.agent._best_child(root), safe)
        hanging.proven = 1.0
        self.assertIs(self.agent._best_child(root), hanging)
        hanging.proven = 0.5
        self.assertIs(self.agent._best_child(root), hanging)

    def test_all_root_moves_remain_available_including_sacrifices(self):
        board = chess.Board("3r2k1/8/8/3p4/8/8/8/3Q2K1 w - - 0 1")
        self.agent.simulations = len(list(board.legal_moves))
        self.agent.rollout_depth = 0
        self.agent.choose_move(board)
        expanded = {row['move'] for row in self.agent.last_root_stats}
        self.assertEqual(expanded, {m.uci() for m in board.legal_moves})
        self.assertLess(next(r['safety_cp'] for r in self.agent.last_root_stats
                             if r['move'] == 'd1d5'), 0)

    def test_timeout_inside_exchange_restores_the_board(self):
        board = chess.Board("r6k/8/8/8/8/8/Q7/6K1 w - - 0 1")
        before = board.fen(), tuple(board.move_stack)
        with patch.object(self.agent, "_exchange_reply", side_effect=_SearchTimeout):
            with self.assertRaises(_SearchTimeout):
                self.agent._move_safety(board, chess.Move.from_uci("g1h1"))
        self.assertEqual((board.fen(), tuple(board.move_stack)), before)

    def test_recorded_queen_blunder_is_avoided_near_fifty_move_limit(self):
        # Same board and halfmove clock as 46.Qa2+ in the requested KQ test.
        board = chess.Board("8/8/8/8/8/1k4K1/8/Q7 w - - 90 46")
        self.agent.simulations = 200
        selected = self.agent.choose_move(board)
        self.assertGreaterEqual(self.agent._move_safety(board, selected), 0)

    def test_feature_can_be_disabled(self):
        with self.assertRaises(ValueError):
            MCTSImproveAgent(exchange_safety=1)
        agent = MCTSImproveAgent(simulations=3, rollout_depth=1, seed=0,
                                 exchange_safety=False)
        with patch.object(agent, "_move_safety", side_effect=AssertionError('disabled')):
            self.assertIn(agent.choose_move(chess.Board()), chess.Board().legal_moves)


if __name__ == '__main__':
    unittest.main()
