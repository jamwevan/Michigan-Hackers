# Stock Market Simulation

This project simulates a real-time electronic stock exchange by processing and matching buy/sell orders according to defined rules. It models traders, stocks, and a time-based transaction system, supporting multiple simulation modes and detailed output logging.

---

## Overview

The simulation reads input data representing trader activity, timestamps, and stock orders. It supports:

- **Modes**: TL (Time Loop), PR (Priority), and TT (Time Travel)
- **Matching Engine**: Uses `priority_queue` with custom comparators to match orders by price and time
- **Output**: Logs trades, unmatched orders, summaries, and optional time traveler analytics

---

## File Structure

```
.
├── README.md                     # Project documentation
├── Makefile                      # Build script
├── main.cpp                      # Main C++ source file
├── getopt.h                      # Header for command-line argument parsing
├── P2random.h                    # Provided header for TL mode randomization

# Input test files
├── large-input-PR.txt
├── large-input-TL.txt
├── small-input-PR.txt
├── small-input-TL.txt
├── spec-input-PR.txt
├── spec-input-TL.txt

# Output verification files
├── output_test.txt
├── large-output-all.txt
├── large-output-i.txt
├── large-output-m.txt
├── large-output-t.txt
├── large-output-v.txt
├── small-output-all.txt
├── small-output-i.txt
├── small-output-m.txt
├── small-output-t.txt
├── small-output-v.txt
├── spec-output-all.txt
├── spec-output-i.txt
├── spec-output-m.txt
├── spec-output-t.txt
├── spec-output-v.txt
```

---

## How to Compile

```bash
make
```

This compiles the source code and generates the `market` executable.

---

## How to Run

Run the simulator using:

```bash
./market -m TL -t 3 < input.txt
```

### Command-Line Flags

| Flag        | Description                                      |
|-------------|--------------------------------------------------|
| `-m MODE`   | Simulation mode (`TL`, `PR`, or `TT`)            |
| `-t NUM`    | Focus output on a specific trader ID             |
| `-v`        | Verbose mode — prints detailed match information |
| `-g`        | Graphical/summary output mode                    |
| `-s`        | Enables time traveler summary                    |

---

## Dependencies

### `getopt.h`

This header provides `getopt()`, which parses command-line flags like `-m TL`, `-v`, and `-t 3`.

### `P2random.h`

This is a provided utility to seed and generate deterministic random input for TL mode. It must be present to run TL mode correctly.

---

## Legal Command Line Examples 

```bash
./market_debug < infile.txt > outfile.txt
./market_debug --verbose --trader_info < infile.txt
./market_debug --verbose --median > outfile.txt
./market_debug --time_travelers
./market_debug --trader_info --verbose
./market_debug --vmit
```

## Illegal Command Line Example 

```bash
./market_debug -v -q
# '-q' is not a recognized flag      
```

---


