"""Correctness gates for progressive expansion and static rollout estimates."""

import math
import random
import unittest
from unittest.mock import patch

import chess

from agents.mcts_improve import MCTSImproveAgent, _Node


class SearchFeatureTests(unittest.TestCase):
    def test_invalid_feature_configuration(self):
        for options in (
            {"progressive_expansion": 1}, {"widening_base": 0},
            {"widening_base": math.inf}, {"widening_exponent": 0},
            {"widening_exponent": 1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                MCTSImproveAgent(**options)

    def test_root_actions_are_all_admitted(self):
        agent = MCTSImproveAgent()
        root = _Node(turn=True, visits=1, untried=[chess.Move.from_uci("a2a3")])
        root.children = [_Node(turn=False) for _ in range(20)]
        self.assertTrue(agent._can_expand(root))

    def test_internal_width_grows_with_visits(self):
        agent = MCTSImproveAgent()
        node = _Node(turn=False, parent=_Node(turn=True), visits=4,
                     untried=[chess.Move.from_uci("a7a6")])
        node.children = [_Node(turn=True) for _ in range(4)]
        self.assertFalse(agent._can_expand(node))
        node.visits = 9
        self.assertTrue(agent._can_expand(node))
        agent.progressive_expansion = False
        node.visits = 1
        self.assertTrue(agent._can_expand(node))

    def test_remaining_defenses_are_never_hidden_by_width_or_proofs(self):
        agent = MCTSImproveAgent()
        node = _Node(turn=False, parent=_Node(turn=True), visits=1,
                     untried=[chess.Move.from_uci("a7a6")])
        node.children = [_Node(turn=True, proven=1.0) for _ in range(4)]
        agent._update_proven(node, True)
        self.assertIsNone(node.proven)
        self.assertTrue(agent._can_expand(node))
        # One newly admitted opponent reply refutes the apparent win.
        node.children.append(_Node(turn=True, proven=0.0))
        agent._update_proven(node, True)
        self.assertEqual(node.proven, 0.0)

    def test_static_win_estimate_is_not_a_node_proof(self):
        agent = MCTSImproveAgent(simulations=2, rollout_depth=0, seed=0)
        with patch.object(agent, "_static_value", return_value=1.0) as estimate:
            agent.choose_move(chess.Board())
        self.assertEqual(estimate.call_count, 2)
        self.assertEqual(agent.last_stats["stop_reason"], "simulations")
        self.assertTrue(all(child["proven"] is None for child in agent.last_root_stats))

    def test_zero_rollout_uses_static_value_at_the_current_leaf(self):
        board = chess.Board()
        before = board.fen(), tuple(board.move_stack)
        agent = MCTSImproveAgent(rollout_depth=0)
        with patch.object(agent, "_static_value", return_value=0.37) as estimate:
            result = agent._rollout(board, chess.WHITE, random.Random(0), None)
        self.assertEqual(result, (0.37, 0, "heuristic_cutoffs"))
        estimate.assert_called_once_with(board, chess.WHITE)
        self.assertEqual((board.fen(), tuple(board.move_stack)), before)

    def test_features_can_be_disabled_and_visits_reconcile(self):
        agent = MCTSImproveAgent(simulations=30, rollout_depth=2, seed=0,
                                 progressive_expansion=False)
        board = chess.Board()
        self.assertIn(agent.choose_move(board), board.legal_moves)
        self.assertEqual(agent.last_stats["progressive_selections"], 0)
        self.assertEqual(agent.last_stats["iterations"], agent.last_stats["root_child_visits"])


if __name__ == "__main__":
    unittest.main()
