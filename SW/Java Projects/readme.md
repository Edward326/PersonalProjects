# Football Game (Console Match Simulator)

## Purpose

A simple text-based football match simulator written in Java, using randomly generated ball
coordinates and Java's checked-exception mechanism to drive the play-by-play commentary. The
project exists in two versions: an initial version (`game_INIT`) and a refined version
(`enchanced_game`) with more realistic ball movement and a time-based match instead of a
fixed number of shots.

## Implementation

### Pitch model
The pitch is modeled as a coordinate grid: `x` in `[0, 100]` represents position along the
length of the pitch (goal lines at `x = 0` and `x = 100`), and `y` in `[0, 50]` represents
position across its width. The goal mouth is the `y` range `[20, 30]`; `y = 0` and `y = 50` are
the touchlines (out of bounds).

### Events as exceptions (`Out`, `Goal`, `Corner`)
Rather than returning a result code, `Ball.shoot(x, y, teams)` is declared to `throw` one of
three custom checked exceptions to signal what happened to the shot:
- **`Out`** — thrown when the ball reaches a touchline (`y == 0` or `y == 50`).
- **`Goal`** — thrown when the ball crosses a goal line (`x == 0` or `x == 100`) while `y` is
  within the goal mouth (`20 <= y <= 30`); the exception carries the scoring team's name.
- **`Corner`** — thrown when the ball crosses a goal line outside the goal mouth; the exception
  carries the defending team's name (i.e. who gets the corner).

The `simulate()` loop in `Game` catches these exceptions (via `instanceof` checks) to update the
running score, corner count and out count, and to print commentary for each event.

### Coordinate generation (`CoordinateGenerator`)
A helper class wraps `java.util.Random` (seeded from the current time) and exposes
`generateX()`/`generateY()` to produce the next ball position each "shot":

- **`game_INIT`** — coordinates are generated close to uniformly at random across the pitch,
  with `x`/`y` snapped to the boundary (`0`/`100` or `0`/`50`) when the raw random value falls
  in a small edge band.
- **`enchanced_game`** — `generateX`/`generateY` additionally take two **team experience**
  parameters (`prob1`, `prob2`, entered by the user, max 50) that bias where the ball is likely
  to end up: higher experience for a team increases the chance the ball is pushed toward the
  opposing goal (and, within `generateY`, increases the chance it lands inside the goal mouth
  rather than wide), producing a less uniformly random and more "skill-weighted" distribution of
  shots than the initial version.

### Match loop (`Game.simulate()`)
Both versions repeatedly generate a new ball position, "shoot" the ball, and react to whichever
exception is thrown, updating and printing the running score
(`teamA-teamB scor: X:Y cornere: X:Y outuri: N`) after each event.

- **`game_INIT`** runs for a fixed number of iterations (`maxshots = 1000`), i.e. the match ends
  after a set number of shot attempts regardless of elapsed "time".
- **`enchanced_game`** instead models **match time**: it loops for `playTime = 90` simulated
  minutes, calling `wait(playTimeUnit * 1000)` (the method is `synchronized`) between shots so
  the simulation plays out at roughly one event per second, and prints the current match minute
  (`currentplayTimeUnit'`) alongside each goal.

### Input
- **`game_INIT`** hardcodes the two team names (`"FC Bihor"` vs `"FCSB Steaua"`) directly in
  `main()`.
- **`enchanced_game`** reads both team names and each team's experience value from standard
  input (via `BufferedReader`/`InputStreamReader`) before starting the simulation, making the
  matchup and skill balance configurable per run.

## Notes
- Both versions are single-file console programs (`javac game.java` /
  `javac enchanced_game.java`, then run the resulting `Game` class); the checked-in `.class`
  files are pre-compiled build artifacts.
- Using exceptions purely for control flow (rather than error handling) is a deliberate design
  choice here, used to practice Java's exception hierarchy and `try`/`catch`/`finally` semantics.
