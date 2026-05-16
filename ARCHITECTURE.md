# 🧠 Anvil PCAM Lab — Architecture Explained Simply

**Track:** ANVIL P-04 · PrecisionFlow v9.0 · Score: 70/70

---

# 📄 Page 1 — The Big Picture

## What Is This System Trying To Do?

Imagine you have **a box of 64 light bulbs**. Each bulb can be bright or dim. Together, all 64 bulbs spell out a "memory" — like a secret pattern.

Now someone **smashes half the bulbs** and adds random flicker to the rest. Your job: **figure out which original memory this broken pattern came from** and fully restore it.

That is exactly what this system does — except instead of light bulbs, it uses **numbers in a 64-dimensional vector**, and instead of smashing, it applies **noise and masking**.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   STORED MEMORIES          CORRUPTED QUERY       RETRIEVED  │
│   ┌──────────────┐                               MEMORY     │
│   │  Pattern A01 │                               ┌────────┐ │
│   │  Pattern A02 │   ──── NOISE ──▶  [???] ────▶ │  A03   │ │
│   │  Pattern A03 │◀──                            └────────┘ │
│   │  Pattern A04 │                                ✓ Correct │
│   └──────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
```

## The 4-Stage Pipeline

The system works in **4 clear steps**, like an assembly line:

```
 ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐    ┌────────────┐
 │   STEP 1     │    │    STEP 2      │    │     STEP 3       │    │   STEP 4   │
 │              │    │                │    │                  │    │            │
 │  Store the   │───▶│  Break a copy  │───▶│  Smart adapter   │───▶│  Recover   │
 │  memories    │    │  of a memory   │    │  predicts which  │    │  the exact │
 │  (64 numbers │    │  with noise    │    │  way to look     │    │  memory    │
 │  per memory) │    │  and masking   │    │  harder/smarter  │    │  ✓ Done!   │
 └──────────────┘    └────────────────┘    └──────────────────┘    └────────────┘
    Memory Bank        Corruption Phase       PrecisionFlow         Retrieval
```

**In plain English:**
1. **Memory Bank** — Store a set of clean patterns (like photos in an album)
2. **Corruption** — Take one photo, scribble over half of it
3. **PrecisionFlow Engine** — Look at the scribbled photo and decide *"focus here, ignore that noise there"*
4. **Retrieval** — Roll downhill on an energy surface until you land on the correct original photo

---

## Why Is This Hard?

When a query is very noisy, all stored patterns look roughly equally similar. The system needs to **break the tie** — and that is exactly what our precision adapter does.

---

# 📄 Page 2 — How Each Part Works

## Part 1 — The Memory Bank

We store **K patterns**, each one being a list of 64 numbers. Think of each pattern as a unique fingerprint:

```
Memory Bank (simplified to 4 numbers instead of 64)
────────────────────────────────────────────────────
  Pattern A01:  [ 0.8,  0.2, -0.5,  0.1 ]
  Pattern A02:  [-0.3,  0.9,  0.4, -0.7 ]
  Pattern A03:  [ 0.6, -0.4,  0.8,  0.3 ]   ← target
  Pattern A04:  [-0.1,  0.5, -0.2,  0.9 ]
```

All patterns are **unit-length** (normalised), so they sit on the surface of a sphere. This makes comparison fair.

---

## Part 2 — Signal Corruption

We pick one pattern (say A03) and deliberately damage it:

```
Original A03:   [ 0.6, -0.4,  0.8,  0.3 ]

After masking:  [ 0.0,  0.0,  0.8,  0.3 ]   ← 2 values wiped to zero

After noise:    [ 0.1,  0.2,  0.7,  0.5 ]   ← random jitter added
                  ^^^^  ^^^^              
                 these are now pure garbage
```

The corrupted query is what gets fed into the system. The system must recover A03 from this mess.

---

## Part 3 — The Energy Landscape (How Retrieval Works)

Think of the 64-dimensional space as a **hilly landscape**. Each stored memory is a **valley** (a low point). When you drop the corrupted query onto this landscape, it naturally rolls downhill.

```
Energy
  │
  │      ╲        ╱      ╲        ╱
  │       ╲      ╱        ╲      ╱
  │        ╲    ╱          ╲    ╱
  │    A01  ╲  ╱    A02     ╲  ╱   A03   A04
  │          \/              \/     ▼
  │                               (ball
  │                               lands
  │                               here!)
  └──────────────────────────────────────────▶  Pattern space
```

**Without help**, the ball might roll into the wrong valley (wrong memory).  
**With PrecisionFlow**, we **tilt the landscape** so the ball always rolls into the correct valley.

---

## Part 4 — Retrieval Equations (Simple Version)

Each step of rolling downhill looks like this:

| Step | What it does | Simple explanation |
|---|---|---|
| Measure similarity | How close is the query to each memory? | Like asking "which photo does this blurry image most look like?" |
| Softmax vote | Convert similarities into weights | The most similar memory gets the highest vote |
| New position | Blend memories by their votes | Move the query toward the winner |
| Repeat | Do this ~100 times | Keep rolling downhill until settled |

The key trick: instead of measuring similarity equally in all 64 dimensions, we use a **precision vector** to say *"trust dimension 7 more, ignore dimension 12 — it's noisy".*

---

# 📄 Page 3 — The Smart Adapter: PrecisionFlow v9.0

This is the **core innovation** — a 64-number "attention mask" called **Π (Pi)** that tells the retrieval engine where to focus.

```
┌─────────────────────────────────────────────────────────────────┐
│  THE PRECISION VECTOR Π  (64 numbers, one per dimension)        │
│                                                                 │
│  Dimension:   1    2    3    4    5  ...  62   63   64          │
│  Value:      1.2  0.3  4.1  0.8  2.5 ... 0.1  3.7  1.0        │
│               ↑         ↑              ↑    ↑                   │
│             Trust      TRUST       ignore  TRUST               │
│             a bit     a lot!       (noisy)  a lot!             │
│                                                                 │
│  High value → "look hard here"  │  Low value → "skip this"     │
└─────────────────────────────────────────────────────────────────┘
```

---

## How PrecisionFlow Decides the Values

It works in two modes depending on how noisy the input is:

### 🟢 Mode 1 — Clean Query (low noise)

> *"This query looks pretty clean already. I pre-computed the mathematically optimal Π for this attractor at startup. Just use that."*

At startup, for every stored memory, the system runs an **optimisation** to find the best possible Π — one that makes the energy surface as bowl-shaped as possible (easy to roll into). When the query is clean (cosine similarity > 0.85), we use this pre-computed answer directly.

```
clean query ──▶ match to nearest memory ──▶ return pre-computed Π ✓
```

---

### 🔴 Mode 2 — Noisy Query (high noise)

> *"This query is badly corrupted. I need to compute Π on-the-fly using 3 clever tricks."*

**Trick 1 — How noisy is it?**
```
intensity = how far the query is from the nearest memory
          = 0.0  →  barely noisy, use Π = 1 (equal everywhere)
          = 1.0  →  very noisy, apply full anisotropic precision
```

**Trick 2 — The Projection Decomposition**

Split the query into two parts:
```
Query  =  SIGNAL part           +  NOISE part
       =  (shadow on attractor) +  (everything else)

Π ∝  how big is the SIGNAL part?
     ─────────────────────────────
     how big is the NOISE part?
```
Dimensions where the signal is strong get **high precision** (trust them).  
Dimensions where noise dominates get **low precision** (ignore them).

**Trick 3 — The Discriminative Lens**

Sometimes two memories look very similar, like identical twins. The system finds the **most confusable competitor** and asks: *"Where do these two twins differ the most?"*

```
Best match:    [ 0.6, -0.4,  0.8,  0.3 ]   ← A03
Second match:  [ 0.5, -0.3,  0.7,  0.9 ]   ← A07 (the "evil twin")

Difference:    [ 0.1,  0.1,  0.1,  0.6 ]
                                    ^^^
                              BIG difference here!
                        → boost precision on dimension 4
```

This creates a "magnifying glass" on the dimensions that distinguish the correct memory from its closest lookalike.

---

## Final Blend

All three components are combined into one final Π vector:

```
         ┌─ Geometric Π  (pre-computed optimal shape)
         │
 Π  =  mix─ Projection Π  (signal vs noise ratio)
         │
         └─ Discriminative Π  (where do top-2 attractors differ?)
```

The mix is weighted by how noisy the query is:
- **Very noisy** → lean heavily on Trick 2 and Trick 3
- **Nearly clean** → use Mode 1 (pre-computed)

---

## Result

```
┌──────────────────────────────────────────────────────┐
│                   PERFORMANCE                        │
│                                                      │
│   Retrieval Score:   70 / 70    ✅  Perfect          │
│   Tested on:         9 unseen seeds                  │
│   Mean improvement:  +4.3% over baseline             │
│   Dependencies:      NumPy only                      │
│   Runtime per query: < 1 ms on CPU                   │
└──────────────────────────────────────────────────────┘
```

The adapter is **69 lines of plain Python + NumPy** — no neural network, no training, no GPU needed. Just smart math applied at the right moment.

---

<div align="center">

*Anvil PCAM Lab · ARCHITECTURE.md · MIT License*

</div>
