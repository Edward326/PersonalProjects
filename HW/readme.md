# ALU_CACHE_UNIT

## Purpose

A two-part Verilog hardware design built and simulated in ModelSim / EDA Playground:

- **ALU-Partea 1** ("Part 1") — a 32-bit Arithmetic Logic Unit supporting addition, subtraction,
  multiplication, division, shifts and bitwise logic operations.
- **Cache-Partea2** ("Part 2") — a 4-way set-associative data cache with true-LRU replacement,
  sitting on top of a simulated main memory.

Both parts were built incrementally: each functional unit was developed and testbenched in its
own folder as a standalone module, then imported into a single top-level `centralUnit.v` in a
`mainframeALU_UNIT` / `mainframeCACHE_UNIT` folder where everything is wired together (as
described in each part's own `structuraTotala.txt` planning notes).

---

## ALU-Partea 1 — Arithmetic Logic Unit

### Purpose
A clocked, 32-bit ALU that executes one of 12 operations selected by a 5-bit opcode, described
in `Docs/Documentation.txt`:

| op | Operation | op | Operation |
|---|---|---|---|
| 0 | N (neutral / idle) | 6 | Shift X right |
| 1 | Sum | 7 | Shift Y left |
| 2 | Difference | 8 | Shift Y right |
| 3 | Multiply | 9 | AND |
| 4 | Divide | 10 | OR |
| 5 | Shift X left | 11 | XOR |

Multiplication and division are multi-cycle operations synchronized to the clock: if too few
clock cycles are supplied, the unit holds the result at 0 to avoid returning a corrupted partial
result; the unit is designed to detect completion internally and stop consuming cycles once the
correct result is ready, even if more cycles than necessary are supplied.

### Implementation

**Adder/Subtractor (`ADDER_SUBTR_UNIT`)** — a hierarchical **Carry-Skip Adder (CSkA)** built from
8-bit **Ripple-Carry Adder** blocks:
- `FAC` — a single full adder (sum/carry from `A`, `B`, `cin`).
- `RCA` — an 8-bit ripple-carry adder chaining 8 `FAC` cells.
- `FAC_Star` / `RCA_Star` — a variant full adder/RCA that additionally computes a **group
  propagate** signal (`pi`), used to let the carry skip ahead across a whole 8-bit group when
  every bit in that group would propagate a carry, avoiding the full ripple-carry delay.
- `CSkA` — chains 4 such 8-bit blocks (2 plain `RCA`, 2 `RCA_Star`) to build a fast 32-bit
  adder, with carry-skip logic between groups (`carry[i] | (Propagate[i] & carry[i-1])`).
  Subtraction reuses the same adder by XOR-ing the second operand with the carry-in
  (two's-complement trick via `makeXor`), so `cin = 1` performs a subtract.
  The module is actually parameterized for **three operand widths at once** (32-bit for
  add/subtract, 33-bit and 34-bit for internal use by the divider and multiplier respectively),
  computing all three sums in parallel from shared sub-adder instances.

**Multiplier (`MULTIPLIER_UNIT`)** — a sequential **radix-8 (modified Booth) multiplier**:
- `controlUnit` is a finite state machine that inspects overlapping 4-bit windows of the
  multiplier (`w,x,y,z`) and decodes them into a Booth digit in `{0, ±1, ±2, ±3, ±4}` per state,
  driving which partial product (`m`, `2m`, `3m`, `4m`, generated via `operations`/`lshift`) gets
  accumulated into the running product each cycle.
- `operations` computes `2m` (shift), `3m` (via the shared `CSkA` adder, `m + 2m`) and `4m`
  (shift) and selects the one needed for the current Booth digit; `lshift` shifts the
  accumulator by 3 bits per step (radix-8) and `fshift` performs a final correction shift.
  A `counter` module tracks how many Booth steps have been performed.
- The top-level `multiplier` module wires the FSM, operand shifter and accumulator together and
  exposes a `suff` ("sufficient"/done) flag once the multi-cycle operation completes.

**Divider (`DIVIDER_UNIT`)** — a sequential **non-restoring division** algorithm:
- `controlUnit` cycles the FSM through: initialize → shift → add/subtract divisor (via `CSkA`,
  based on the current remainder's sign) → conditionally set the quotient bit
  (`checkNotRestore`) → shift again → repeat until the counter reaches 31 iterations, then stop.
- `rshift` performs the combined shift-left of the remainder/quotient register pair each
  iteration; `counter` tracks the iteration count (fixed at ~165 clock cycles per the
  documentation, regardless of operand values, since non-restoring division always runs a fixed
  number of iterations for a given bit width).

**Logic unit (`LOGICop_UNIT`)** — bitwise `AND`, `OR`, `XOR` (each built with a `generate` loop
over all 32 bits, sign-extended into the ALU's common 67-bit result width) plus arithmetic
left/right shifts of `X` and `Y` by 32 bits, all computed combinationally and in parallel on the
same clock edge, per `strucura.txt`'s design note that all four logic-unit outputs must be
available simultaneously.

**Top-level ALU (`mainframeALU_UNIT/centralUnit.v`)** — instantiates the adder/subtractor,
multiplier, divider and logic unit in parallel, decodes the 5-bit `op` input to enable exactly
one of the multi-cycle units (multiply/divide) and to drive an 11-to-1 output multiplexer
(`mux11to1`) that selects which unit's result is presented at the ALU's output, gated by each
unit's own "done" (`suff`) signal so an in-progress multi-cycle result is never read out early.

**Testing** — every unit has its own self-checking testbench (`*_tb` modules with `$monitor` /
fixed stimulus and cycle counts) usable standalone in `ModelSim` (`.mpf` project files,
`run_*.txt` transcripts) or in EDA Playground; `alu_tb` in the top-level file exercises the fully
assembled ALU.

---

## Cache-Partea2 — Data Cache

### Purpose
A 4-way set-associative data cache sitting between a CPU-style requester and (a simulated) main
memory, supporting single-byte reads and writes with true-LRU replacement on misses. Built to a
10-day plan (see `structuraTotala.txt`): FSM first, then hit-detection, then the miss/eviction/
allocation states, then write-hit, then testing.

### Address layout
A 20-bit `adressWord` is split as:

| Bits | Field | Meaning |
|---|---|---|
| `[19:13]` | tag | 7-bit tag compared against each way's stored tag |
| `[12:6]` | index | 7-bit index selecting one of 128 cache sets/lines |
| `[5:3]` | block offset | selects 1 of 8 64-bit words within the 64-byte line |
| `[2:0]` | word offset | selects 1 of 8 bytes within the selected 64-bit word |

### Cache storage
Four parallel banks (`cacheBANK_A..D`), each 128 entries deep, model the 4 ways of a set. Each
entry is a 523-bit record packed as: bit `[0]` valid bit, bits `[3:2]` a 2-bit **age** counter
(true LRU: 0 = most recently used, 3 = least recently used / eviction candidate), bits `[10:4]`
the stored tag, and bits `[522:11]` a 512-bit (64-byte) data block.

### Implementation
- **`controlUnit`** — the cache's FSM: on `EN`, it dispatches to a read or write path
  (`W_R`), passes through a hit/miss decision state, and for a write updates the block *in
  place* before continuing; for a miss it branches on whether the target set is already full
  (`CacheFULL`) before allocating a new line.
- **`selectWord` / `detectBlockInCache`** — compare the incoming tag against all 4 ways at the
  addressed index in parallel (`comparator` x4) to determine hit/miss and, on a hit, which way
  (`SET`) matched; also computes whether the whole set is full (all 4 valid bits set) and
  extracts the addressed byte from the addressed way via a `mux4to1` → `mux8to1A` → `mux8to1B`
  selection chain (block → word → byte).
- **`ReadHit` / `WriteHit`** — perform the actual read or in-place write of the addressed byte,
  and update the LRU state: every way in the set that was more recently used than the accessed
  way has its age incremented, and the accessed way's age is reset to 0 (a standard LRU stack
  update using per-way age counters instead of a full recency list).
- **`MissCaseNFULL` / `MissCaseFULL`** — on a miss, age every valid way in the set by one
  (`MissCaseNFULL` only up to the target way when the set isn't full yet, `MissCaseFULL` for all
  4 ways when the set is already full and a victim must be evicted), preparing the target way to
  become the new "most recently used" entry once it's filled.
- **`Allocate`** — implements the miss-fill: since there is no real main-memory model in this
  simulation, missing 64-byte blocks are filled with **random data** (`$urandom`) as a stand-in
  for a memory fetch, and the way's valid bit and tag are set to reflect the newly cached line.
- **`centralUnit`** — the top-level module: holds the 4 cache-bank register arrays, instantiates
  the FSM and the pipeline of combinational stages above (detect → read-hit → miss-not-full →
  miss-full → allocate → write-hit), and commits the resulting bank contents on each clock edge.

### Testing
`centralUnit_tb` drives a sequence of read/write requests designed to exercise every case: cold
misses filling all 4 ways of a set, a read hit, and a full-set LRU eviction (verifying that the
least-recently-used way — the one with age `3` — is the one replaced), followed by a write-hit
and a read-back of the written byte. Timing notes in the testbench record the expected latency:
roughly 5 clock cycles for a cache hit and roughly 10 clock cycles for a miss (fetch + fill +
completion), with 1 clock cycle modeled as 2 simulation levels of 10ns each.

---

## Notes
- Both designs were developed and simulated with **ModelSim** (`.mpf` project files, `work/`
  compiled-library directories, `.wlf` waveform logs) and are also noted as compatible with
  **EDA Playground**.
- `mainframeALU_UNIT` and `mainframeCACHE_UNIT` each contain a full copy of their part's
  sub-units concatenated into one file — this is the buildable, self-contained version of each
  design; the individual `*_UNIT` folders are the original per-module development/test
  sandboxes.
- Presentation slides for both parts are included under each `Docs/Prezentare.pptx`.
- The ALU and cache are currently independent designs (the cache does not yet consume the ALU's
  address-generation output); integrating them into a single pipeline would be a natural next
  step.
