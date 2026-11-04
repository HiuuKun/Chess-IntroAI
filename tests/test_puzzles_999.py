"""History-counter regression and public CLI/worker integration checks."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess

from agents import AGENT_FACTORIES
from agents.mcts_agent import MCTSAgent
from agents.random_agent import RandomAgent
from benchmarks import reference
from benchmarks.puzzles_999 import DEFAULT_DATASET, board_for, load_dataset, normalize_puzzle
from benchmarks.run_999 import _worker, agent_options, get_factory, main, parser_for_cli, run_trial


class ScriptedAgent:
    def __init__(self, moves):
        self.moves = iter(moves)

    def choose_move(self, board, time_limit=None):
        return chess.Move.from_uci(next(self.moves))


class Connection:
    def send(self, result):
        self.result = result

    def close(self):
        self.closed = True


class Puzzle999Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = load_dataset()
        cls.puzzles = cls.payload["puzzles"]
        cls.raw = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8-sig"))

    def test_full_file_validates_and_has_999_entries(self):
        self.assertEqual(self.payload["metadata"]["puzzle_count"], 999)
        self.assertEqual(self.payload["metadata"]["categories"],
                         {"mate_in_one": 333, "mate_in_two": 333, "mate_in_three": 333})
        self.assertEqual(self.payload["metadata"]["duplicate_positions"], 0)

    def test_counter_corrections_survive_draw_checks_and_push_pop(self):
        corrections = self.payload["metadata"]["counter_corrections"]
        self.assertEqual([p["source_index"] for p in corrections], [565, 903, 911])
        for index in (565, 903, 911):
            with self.subTest(index=index):
                puzzle = self.puzzles[index - 1]
                board = board_for(puzzle)
                before = board.fen(), tuple(board.move_stack)
                for _ in range(2):
                    board.outcome(claim_draw=True)
                    move = next(iter(board.legal_moves))
                    board.push(move)
                    board.pop()
                    self.assertEqual((board.fen(), tuple(board.move_stack)), before)

    def test_bad_position_and_inconsistent_san_are_rejected(self):
        record = copy.deepcopy(self.raw[0])
        record["FEN"] = chess.STARTING_FEN
        with self.assertRaisesRegex(ValueError, "PGN position"):
            normalize_puzzle(record, 1)
        record = copy.deepcopy(self.raw[0])
        record["MovesSAN"] = "a3"
        with self.assertRaises(ValueError):
            normalize_puzzle(record, 1)

    def test_original_factory_accepts_budget_without_registry_changes(self):
        original = AGENT_FACTORIES["MCTS"]
        self.assertIs(get_factory("MCTS"), MCTSAgent)
        self.assertIs(AGENT_FACTORIES["MCTS"], original)
        args = parser_for_cli().parse_args(["--simulations", "200"])
        self.assertEqual(agent_options("MCTS", args, {}), {"simulations": 200})

    def test_custom_import_and_improved_constructor_options(self):
        self.assertIs(get_factory("agents.random_agent:RandomAgent"), RandomAgent)
        args = parser_for_cli().parse_args([])
        options = agent_options("mcts_improve", args, {})
        self.assertEqual(agent_options("mcts_improve", args, {}), {"simulations": 200})
        self.assertEqual(get_factory("mcts_improve")(**options).rollout_depth, 30)
        args.rollout_depth = 0
        self.assertEqual(agent_options("mcts_improve", args, {}),
                         {"simulations": 200, "rollout_depth": 0})

    def test_json_seed_and_unsupported_explicit_option_fail(self):
        args = parser_for_cli().parse_args([])
        with self.assertRaisesRegex(ValueError, "--seeds"):
            agent_options("mcts_improve", args, {"seed": 7})
        args.rollout_depth = 2
        with self.assertRaisesRegex(ValueError, "does not accept"):
            agent_options("Random", args, {})

    def test_worker_reuses_scoring_and_restores_global_adapters(self):
        puzzle = self.puzzles[564]
        previous_loader = reference.board_for
        previous_factory = AGENT_FACTORIES["MCTS"]
        connection = Connection()
        factory = lambda: ScriptedAgent(puzzle["solution_uci"][::2])
        with patch("benchmarks.run_999.get_factory", return_value=factory):
            _worker(connection, "MCTS", puzzle, 0, {})
        self.assertEqual(connection.result["status"], "passed")
        self.assertEqual(connection.result["terminal_reason"], "CHECKMATE")
        self.assertTrue(connection.closed)
        self.assertIs(reference.board_for, previous_loader)
        self.assertIs(AGENT_FACTORIES["MCTS"], previous_factory)

    def test_hard_timeout_returns_a_record(self):
        result = run_trial("MCTS", self.puzzles[0], 0, {"simulations": 100000}, .001)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["puzzle_id"], "p0001")

    def test_public_cli_runs_both_agents_and_records_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--agent", "MCTS", "mcts_improve", "--simulations", "2",
                             "--limit", "1", "--workers", "2", "--output", str(output)])
            self.assertEqual(code, 0)
            rows = [json.loads(s) for s in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({r["agent"] for r in rows}, {"MCTS", "mcts_improve"})
            self.assertTrue(all(r["agent_config"]["simulations"] == 2 for r in rows))
            metadata = json.loads((output / "summary.json").read_text(encoding="utf-8"))["metadata"]
            self.assertTrue(metadata["complete"])
            self.assertEqual(metadata["selected_puzzle_count"], 1)
            self.assertTrue((output / "summary.md").exists())

    def test_invalid_budget_is_rejected_before_starting_workers(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            main(["--simulations", "0"])
        self.assertEqual(exc.exception.code, 2)

    def test_interrupt_preserves_an_incomplete_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "interrupted"
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    "benchmarks.run_999.concurrent.futures.as_completed", side_effect=KeyboardInterrupt):
                code = main(["--limit", "1", "--workers", "1", "--output", str(output)])
            self.assertEqual(code, 130)
            metadata = json.loads((output / "summary.json").read_text(encoding="utf-8"))["metadata"]
            self.assertFalse(metadata["complete"])
            self.assertTrue(metadata["interrupted"])


if __name__ == "__main__":
    unittest.main()
