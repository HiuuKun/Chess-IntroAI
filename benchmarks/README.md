# 999-puzzle agent benchmark

Run from the `Chess-IntroAI` repository root with Python 3.10 or newer:

```powershell
python3.12 -m pip install -r requirements.txt

# Validate every puzzle and reference solution.
python3.12 test_agent_999.py --validate-only

# Compare agents on all 999 puzzles.
python3.12 test_agent_999.py --agent MCTS mcts_improve --simulations 200

# Test the random agent.
python3.12 test_agent_999.py --agent Random

# Quick check before a complete run.
python3.12 test_agent_999.py --agent MCTS mcts_improve --limit 3

# Test a future agent without changing the game registry.
python3.12 test_agent_999.py --agent agents.my_agent:MyAgent --simulations 200
```

[Detailed runner guide](README_999.md). The runner also works as
`python3.12 -m benchmarks.run_999`.

The dataset is [data/puzzles_999.json](data/puzzles_999.json), containing 333
mate-in-one, 333 mate-in-two and 333 mate-in-three puzzles. Every FEN, PGN
history, SAN/UCI solution and final checkmate is validated before running.

The agent plays against recorded opponent replies. Reference solutions and
immediate solver mates pass. A different legal move is `reference_deviation`,
not a game loss. Timeouts, illegal moves and input mutation are reported
separately. Results, settings and source hashes are saved under ignored
`benchmark_results/`, or a new directory selected with `--output`.

The supplied data was preserved byte-for-byte from the development checkout's
`tests/1000_puzzles_in_pgn_san.json`; that old filename contained 999 records.
SHA256: `c963dbec96a9622262a92d865850e04d9e1787b2295d9e726f532bcb8fed541f`.
