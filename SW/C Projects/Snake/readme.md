# Snake

## Purpose

A terminal-based Snake game written in C, targeting a Linux terminal. The project is organized
as a small modular "engine": window/terminal sizing, a game-state-to-2D-grid mapper, and a
text/ANSI-escape-code UI layer, intended to be driven by a central game loop. As committed, the
core gameplay loop (movement, input handling, collision detection) is **unfinished** — the
supporting modules (window sizing, grid mapping, screen rendering) are implemented and
individually testable, but `GameInterface.c`'s `StartGame()` — meant to tie everything together —
is currently an empty stub, and it includes a `MovingMechanism.h` header that doesn't exist yet
in the source tree.

## Implementation

The code is split into focused modules, each with its own header:

### `snakestructure.h` — shared data model
Defines the core types used everywhere else: `foodblock` and `bodyblock` (simple X/Y coordinate
pairs) and `Snake` (a `bodylength`, a dynamic array of `bodyblock` for the body, and a `head`
pointer), plus the display symbols for food (`'O'`) and body segments (`'X'`).

### `WindowInterface.h` / `WindowInterface.c` — terminal geometry
`setPlayWindowLimits()` queries the real terminal size with `ioctl(TIOCGWINSZ)` and derives a
play area (`Window` struct: `upline`, `downline`, `leftline`, `rightline`) that leaves room below
the grid for the score/HUD text.

### `Mapping.h` / `Mapping.c` — game-state to grid
Translates the abstract game state into a 2D `int**` grid that the UI layer can print directly:
- `MakeMatrix()` allocates a `downline x rightline` grid and zero-initializes it.
- `ZeroMapping()` clears the grid.
- `MapMoves()` re-marks the grid from scratch every frame: `0` for empty cells, `1` for any
  snake body/head cell, `2` for food cells, with bounds checks against the play window before
  writing each cell.
- `displayMap()` prints the grid to the terminal, translating `0/1/2` into blank space, the snake
  symbol, or the food symbol.

### `UXDesign.h` / `UXDesign.c` — screens and HUD
- `InGameScreen()` renders the current grid plus a colored (ANSI escape codes) HUD showing the
  score, elapsed play time (via a small `min:sec` `clock` helper), and the control scheme
  (WASD / arrow keys).
- `ExitScreen()` renders a "GAME OVER" screen, compares the final score against a persisted
  high score in `score.dat` (reading/writing it directly with `fread`/`fwrite`), and reports
  whether the high score was beaten.
- `getHiddenChar()` reads a single keypress without echoing it or waiting for Enter, by toggling
  the terminal's `ICANON`/`ECHO` flags via `termios` — the mechanism intended for real-time
  directional input.
- `UXDesign.c` includes its own `main()` as a manual test harness that exercises
  `InGameScreen()` and `ExitScreen()` back-to-back with sample scores/times.

### `GameInterface.h` / `GameInterface.c` — intended entry point (incomplete)
Declares `StartGame()`, documented in a Romanian comment block describing the intended game loop
(spawn food at a random empty cell, move the snake, check for game-over/food-eaten each frame,
re-render, sleep between frames, then show the exit screen and update the high score) — but the
function body is currently empty, so this module is a placeholder for wiring the other pieces
together with actual input handling and collision/movement logic.

### `TEST/test.c` and `run.c` / `auxshifting.c` — early prototypes
- `TEST/test.c` is an isolated experiment for reading arrow-key escape sequences
  (`getHiddenChar()` returning raw byte codes 65/66/67/68 for up/down/right/left) — the input
  method later intended for real gameplay.
- `run.c` / `auxshifting.c` are earlier, standalone prototypes of a horizontally scrolling
  animation (an early stand-in for snake movement) drawn directly with `printf`, predating the
  modular `Window`/`Mapping` design above; they are not part of the current architecture.

## Notes
- Rendering relies on ANSI escape codes for color and cursor movement, and on `ioctl(TIOCGWINSZ)`
  for terminal size, so the game is intended to run in a real Linux terminal.
- `score.dat` is a raw binary file (a single `int`) used to persist the high score between runs.
- To finish the game, `GameInterface.c` needs an implementation of `StartGame()` that ties
  together `WindowInterface`, `Mapping`, `UXDesign` and a (currently missing) movement/collision
  module, using `TEST/test.c`'s input-reading approach for controls.
