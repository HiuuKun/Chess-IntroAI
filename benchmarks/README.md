# Reusable agent benchmarks

Run from the `Chess-IntroAI` repository root with Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt

# Validate the complete datasets without running games.
python test_agent_999.py --validate-only
python test_agent_kq.py --validate-only

# Compare both agents on all 999 mating puzzles, 200 simulations per search.
python test_agent_999.py --agent MCTS mcts_improve --simulations 200

# Quick puzzle smoke test.
python test_agent_999.py --agent MCTS mcts_improve --limit 3

# KQ versus K: full games against Stockfish, highest skill by default.
python test_agent_kq.py --agent MCTS mcts_improve --simulations 200

# Exact requested position: agent Qd2, Ka2; defender Ke6; White moves first.
python test_agent_kq.py --position-ids requested_d2_a2_e6 --colors white

# Quick KQ control check, including both colors.
python test_agent_kq.py --position-ids mate_in_one_control

# Future agents need only the choose_move interface.
python test_agent_999.py --agent agents.my_agent:MyAgent --simulations 200
python test_agent_kq.py --agent agents.my_agent:MyAgent --simulations 200
```

The entry points also work as `python -m benchmarks.run_999` and
`python -m benchmarks.run_kq`. New agents may be registered in
`agents.AGENT_FACTORIES` or referenced by `module:Class`. Interactive agents
are rejected. Constructors accepting `simulations`, `rollout_depth`,
`exploration`, and `seed` receive the selected settings. Other agents retain
parameters they do not expose. Use `--options-file options.json` for additional
constructor options, for example:

```json
{"mcts_improve": {"exchange_safety": false}}
```

## 999 mating puzzles

[Detailed runner guide](README_999.md). The unchanged supplied dataset is
[data/puzzles_999.json](data/puzzles_999.json): 333 mate-in-one, 333 mate-in-two,
and 333 mate-in-three records. Every FEN, PGN history, SAN/UCI solution and final
mate is validated before a run. The runner corrects three history move counters
in memory; it never changes source records.

The opponent follows recorded puzzle replies. A full reference solution or
immediate solver checkmate passes. `reference_deviation` is a different legal
move, not a game loss. Puzzle solution counts are not game win rates.

The data was copied byte-for-byte from the previously supplied
`tests/1000_puzzles_in_pgn_san.json` in the development Chess-framework checkout.
Despite that old filename, it contains 999 records. SHA256:
`c963dbec96a9622262a92d865850e04d9e1787b2295d9e726f532bcb8fed541f`.
No original puzzle IDs or verified publication/license attribution were supplied;
this port preserves the user-provided data without inventing source links.

## King and queen versus king

[data/king_queen_vs_king.json](data/king_queen_vs_king.json) contains four
constructed positions: the exact requested Qd2/Ka2/Ke6 board, an open-board
position, a central defending king, and a mate-in-one control. By default each
agent plays the queen side with both colors, with the board mirrored for Black.
The queen side always moves first. General positions and controls are reported
separately.

Defaults: 200 simulations, seed 42, Stockfish Skill Level 20,
`UCI_LimitStrength=false`, one engine thread, 32 MB hash, 0.2 seconds per
Stockfish move. This is highest skill under that thinking-time limit; no
tablebases are required or configured. Use `--stockfish PATH` for a different
binary. Otherwise the existing Stockfish agent's binary discovery finds the
bundled Windows executable, environment setting or PATH installation.

Useful controls:

- `--stockfish-time 0.5`, `--stockfish-skill 20`
- `--seeds 0 1 2`, `--colors white black`, `--workers 2`
- `--position-ids requested_d2_a2_e6 open_board`
- `--rollout-depth 30 --exploration 1.4`
- `--timeout 60`: hard seconds for agent startup or one response
- `--game-timeout 600`: whole-game wall budget, checked between engine calls
- `--time-limit 0.2`: optional cooperative agent time limit
- `--max-plies 200`: stop an unfinished game at this limit

Agents run in persistent isolated processes, so seeding and agent state are
preserved throughout each game. The parent owns Stockfish and closes it even
when an agent times out. Legal moves and unchanged input history are checked.
An independent replay checks every completed record's SAN, FENs and result.

Claims for repetition and the fifty-move rule are accepted automatically when
available (`claim_draw=True`), matching the framework. Limits, timeouts and
errors remain **unresolved**; they are never counted as draws or losses. A lone
king cannot checkmate, so this suite measures conversion of the material
advantage, queen losses, stalemates and repetition rather than general playing
strength.

## Saved results

Runs create new folders in ignored `benchmark_results/`, never overwriting an
existing folder. Both suites save settings, seeds, source hashes and summaries.
Puzzle runs save per-trial JSONL records; KQ runs additionally save each game's
PGN, move log, full result record and replay audit. `--output PATH` chooses a
new output folder. Historical experiment outputs are not copied into this branch.

## Port validation

Before the unit-test folder was removed, the port passed 64 unit/integration
tests. The benchmark runners and datasets remain self-contained. All 999
puzzle records and their solutions validate. Public-CLI smoke checks passed six puzzle trials, four
mate-in-one KQ games across both agents/colors, and two limited games containing
real Stockfish replies. Limited games remained unresolved. The complete
999-puzzle agent evaluation and complete general-position KQ tournament were
not run during this port.
