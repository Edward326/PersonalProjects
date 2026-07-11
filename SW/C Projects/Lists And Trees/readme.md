# Lists And Trees

## Purpose

A collection of small, self-contained C data-structure libraries and demo programs built while
studying dynamic data structures: singly/doubly linked lists, dynamic arrays, and stacks/queues.
Each subfolder is an independent library with its own header/implementation split, plus a `main`
or `test.c` exercising it. (Despite the folder name, there is currently no tree implementation —
the folder groups the linked-structure exercises done over the same period.)

## Implementation

### `Array Lib` — dynamic array utilities
- **`vectors.h` / `vectors.c`** — a single `fill()` function that allocates an `int` array of a
  given size and fills it with random values in `[0, nmax)`, guarding against re-filling an
  already-allocated array.
- **`file.c`** — a small demo `main` that reads `nelem` and `nmax` from stdin, builds the array
  with `fill()`, and prints it.
- **`va_listCARACT.c`** — a standalone experiment with C variadic functions (`stdarg.h`): a
  `fill_arrays`-style function accepts a format string such as `"ssif"` describing the type of
  each following argument (`s` = string, `i` = int, `f` = float) and sorts the variadic arguments
  into separate typed output arrays, which `print_funct()` then displays.

### `Lists Lib` — singly/doubly linked list libraries
Two successive versions of the same library, plus an applied exercise built on top of it:

- **`lists_library_v1.0.lib`** — the first implementation. `simple_list`/`double_list` wrap
  singly- and doubly-linked chains of `elemS`/`elemD` nodes; the API (`initsimple`, `addsimple`,
  `deletesimple`, `sortsimple`, `verifysortedsimple`, `addreverse_simple`,
  `verifyreversed`, and doubled-linked counterparts) mostly returns/receives raw pointers.
- **`lists_library_v2.0`** — a revised API that switches to consistent double-pointer
  (`elemS **`, `simple_list **`) parameters for any operation that can change what the caller's
  pointer refers to (`new_elemS`, `initsimple`, `trash_simple`, etc.), making memory ownership
  and reallocation safer. The shared struct definitions were also pulled out into their own
  header, `list_structure.h`, so multiple libraries can reuse the same node/list layout. Supports
  insertion, sorted insertion, deletion, sortedness checking, in-place sorting, list reversal, and
  palindrome checking, for both simple and double lists.
- **`applied_lists`** — a small assignment ("Lucrare2") built on the list library: each node
  (`elemS`) carries **two** "next" pointers — `next` (original insertion order) and `nextsorted`
  (sorted order) — so the same set of nodes can be viewed both as originally entered and fully
  sorted, without duplicating data.
  - `main.c` reads integers from a file into such a list, sorts it via `sortsimple()` (bubble
    sort performed over the `nextsorted` chain, counting the number of node accesses), then lets
    the user search for a value with `findnr()`, showing its position in both the unsorted and
    sorted views (with a colored `^` marker under the matching value), and finally deletes it from
    both chains.
  - `listsop_show_multiple_occ/` is a variant of `listsop.c` for handling values that occur more
    than once in the list.

### `Stack_Queue Lib` — stack and queue over a singly linked list
A single `test.c` implementing both structures on the same `nodS` linked-node type:
- **Queue (FIFO)** — `enqueue()` appends to the tail (walking the list to find it),
  `dequeue()` removes from the head, `free_queue()` drains and frees the whole queue.
- **Stack (LIFO)** — `push()` prepends a new head node and returns the updated head pointer,
  `pop()` removes the head node, `free_stack()` drains and frees the whole stack.
- `main()` demonstrates both structures end-to-end: enqueuing/dequeuing a small integer array,
  then pushing/popping the same values, printing the structure's contents after every operation.

## Notes
- These are educational/from-scratch implementations (no dependency on any container library),
  written to practice pointer manipulation, dynamic memory management and API design
  (single- vs. double-pointer interfaces) in C.
- Files ending in `~` throughout the folder are editor backup files and are not part of the
  buildable source.
