def solve_sudoku(sudoku):
    global total_to_solve
    total_to_solve = sum([1 if c == 0 else 0 for r in sudoku for c in r])

    def solve(r1, c1):
        global total_to_solve
        row_list = [c for c in sudoku[r1] if c != 0]
        column_list = [r[c1] for r in sudoku if r[c1] != 0]
        square_list = [
            c
            for i, r in enumerate(sudoku)
            if i // 3 == r1 // 3
            for j, c in enumerate(r)
            if j // 3 == c1 // 3 and c != 0
        ]

        remaining = (
            set(range(1, 10)) - set(row_list) - set(column_list) - set(square_list)
        )
        if len(remaining) == 1:
            total_to_solve -= 1
            sudoku[r1][c1] = list(remaining)[0]

    current_total = total_to_solve
    while total_to_solve > 0:
        for r in range(0, 9):
            for c in range(0, 9):
                if sudoku[r][c] == 0:
                    solve(r, c)
        if current_total == total_to_solve:
            print("Stuck!")
            break
        else:
            current_total = total_to_solve
    return sudoku


def print_sudoku(sudoku):
    print()
    for i, r in enumerate(sudoku):
        print(
            " ".join(
                f" {c}  |" if i in (2, 5) else f" {c}"
                for i, c in enumerate(["*" if c == 0 else str(c) for c in r])
            )
        )
        if i in (2, 5):
            print("-" * 33)
    print()


test_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]


# commands used in solution video for reference
if __name__ == "__main__":
    print_sudoku(test_puzzle)
    solution = solve_sudoku(test_puzzle)
    print_sudoku(solution)
