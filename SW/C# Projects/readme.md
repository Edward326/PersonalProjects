# Cruce (Romanian Schnapsen variant)

## Game Rules

### Match Structure
The game runs for a single hand — play continues until every player has laid down all 6 of
their cards.

In a full game, play continues across multiple hands until one player reaches a target score
agreed at the start of the game. Points accumulate hand after hand. Valid target scores are
**6, 11, 15, or 21** points.

A **turn** consists of applying the relevant action (whichever action is due) to each player in
turn, going counter-clockwise.

### Dealing
Since there are no prior hands to determine a winner who could deal first, a random player is
chosen to deal the cards.

The player to the dealer's right decides how the deck is reorganized before dealing — either by
**cutting** or by **riffling** — and the cards are dealt according to the chosen method.

### Bidding
After dealing, the next phase is the **bidding**, which determines who leads the bidding and
therefore sets the trump suit. Whoever wins the bidding plays the first card of the hand, and its
suit becomes the **trump suit**.

### Playing Tricks
Play then proceeds in tricks:
- The first trick is led by the player to the right of whoever won the bidding.
- Every subsequent trick is led by the winner of the previous trick.

On each trick, every player chooses one card to play from their hand.

Once a trick has been played out, its winner (the trick's leader for the next round) collects
all the cards from the table into their own **`wonCards`** pile.

This repeats for **6 tricks** total: there are 24 cards in total, 4 players, 6 cards each, and
4 cards are played per trick — 24 / 4 = 6 tricks.

### Announcements (20 / 40)
Each player may have the option to announce that they hold a **20** or a **40** combination,
to be scored on the trick they are about to play.

An announcement only becomes **valid** — meaning the announced 20/40 bonus is actually added to
the final score — if:
1. That player wins the current trick, **and**
2. The first card played on that trick belongs to the announced 20/40 pair.

**When an announcement may be made** — only when the hand is not already in its final round
(round/trick 6), and only for a player who is either:
1. The winner of the previous trick, provided they did not already announce on that previous
   trick, or
2. Any player other than the current trick's eventual winner.

### Scoring

**`totalPoints`** — computed at the end of the hand for each player by summing the point values
of every card in that player's `wonCards` pile, plus any points from valid 20/40 announcements.

Note: the player who won the bidding is treated differently from the others — if they fail to
reach the point total they declared during bidding, they are penalized.

**`noPoints`** — a normalized score computed per player as `totalPoints / 33`.

### Determining the Winning Team
`noPoints` is summed per team: player[0] + player[1] form one team, and player[2] + player[3]
form the other. The team with the higher combined score wins the hand.
