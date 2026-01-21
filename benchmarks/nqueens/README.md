# The N-Queens problem

### Formulation

The problem consists in placing `N` chess queens on a $N \times N$ chessboard so that no two queens attack each other; thus, a solution requires that no two queens share the same row, column, or diagonal.

### Configuration options

```
./main_nqueens.out {...}
```

where the available options are:
- **`--N`**: number of queens
  - any positive integer (`14` by default)

- **`--g`**: number of safety check(s) per evaluation (unstable)
  - any positive integer (`1` by default)
