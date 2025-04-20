# Stock Market Simulation

This project simulates a real-time electronic stock exchange by processing and matching buy/sell orders according to defined rules. It models traders, stocks, and a time-based transaction system, supporting multiple simulation modes and detailed output logging.

---

## Installation

```bash
git clone https://github.com/jamwevan/Michigan-Hackers.git
cd Michigan-Hackers/Stock\ Market\ Simulator/
```

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
make market
```

This compiles the source code and generates the `market` executable.

---

## How to Run

Run the simulator using:

```bash
./market < infile.txt > outfile.txt
```

### Command-Line Flags

| Short Flag | Long Flag         | Description                                            |
|------------|-------------------|--------------------------------------------------------|
| `-v`       | `--verbose`       | Enables verbose output                                 |
| `-m`       | `--median`        | Outputs median transaction prices                      |
| `-i`       | `--trader_info`   | Outputs trader-specific order summary                  |
| `-t`       | `--time_travelers`| Outputs time traveler profit-maximizing trade summary  |

---

## Dependencies

### `getopt.h`

This header provides `getopt()`, which parses command-line flags.

### `P2random.h`

This is a provided utility to seed and generate deterministic random input for TL mode. It must be present to run TL mode correctly.

---

## Legal Command Line Examples 

```bash
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


