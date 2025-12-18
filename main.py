import sys
import re
import itertools
import copy
import argparse


# Global debug flag
debug = False


def press_enter_to_continue():
    print("Press Enter to continue...")
    input()


def elves_dial_program():
    start_position = 50
    n_times_pointing_zero = 0

    filename = "data/elves_rotations.txt"

    with open(filename, "r") as file:
        for line in file:
            rotation = line.strip()
            direction = rotation[0]
            steps = int(rotation[1:])

            if direction == "L":
                start_position = (start_position - steps) % 100
            elif direction == "R":
                start_position = (start_position + steps) % 100
            else:
                print(f"Invalid direction '{direction}' in line: {line}")
                continue

            if start_position == 0:
                n_times_pointing_zero += 1

    print(f"The dial pointed to 0 a total of {n_times_pointing_zero} times.")

    press_enter_to_continue()


def elves_dial_program_2():
    start_position = 50
    n_times_pointing_zero = 0

    filename = "data/elves_rotations.txt"
    with open(filename, "r") as file:
        for line in file:
            rotation = line.strip()
            direction = rotation[0]
            steps = int(rotation[1:])

            if direction == "L":
                while steps > 0:
                    start_position = (start_position - 1) % 100
                    steps -= 1
                    if start_position == 0:
                        n_times_pointing_zero += 1
            elif direction == "R":
                while steps > 0:
                    start_position = (start_position + 1) % 100
                    steps -= 1
                    if start_position == 0:
                        n_times_pointing_zero += 1

            else:
                print(f"Invalid direction '{direction}' in line: {line}")
                continue

    print(
        f"The dial pointed to 0 a total of {n_times_pointing_zero} times using method 0x434C49434B."
    )

    press_enter_to_continue()


def invalid_ids_program():
    invalid_ids_running_sum = 0
    invalid_ids_filename = "data/invalid_ids.txt"

    def is_invalid_id(id):
        str_id = str(id)
        pattern = r"^(.*?)\1$"
        match = re.search(pattern, str_id)
        if match:
            return True

        return False

    with open(invalid_ids_filename, "r") as file:
        line = file.readline()
        ranges_list = line.split(",")
        for r in ranges_list:
            start, end = map(int, r.split("-"))
            for i in range(start, end + 1):
                if is_invalid_id(i):
                    invalid_ids_running_sum = invalid_ids_running_sum + i

    print(f"The sum of all invalid IDs is: {invalid_ids_running_sum}")
    press_enter_to_continue()


def invalid_ids_program_2():
    invalid_ids_running_sum = 0
    invalid_ids_filename = "data/invalid_ids.txt"

    def is_invalid_id(id):
        str_id = str(id)
        pattern = r"^([0-9]{1,})\1{1,}$"
        match = re.search(pattern, str_id)
        if match:
            if debug:
                print(f"Invalid ID found: {str_id}")
            return True

        return False

    with open(invalid_ids_filename, "r") as file:
        line = file.readline()
        ranges_list = line.split(",")
        for r in ranges_list:
            start, end = map(int, r.split("-"))
            for i in range(start, end + 1):
                if is_invalid_id(i):
                    invalid_ids_running_sum = invalid_ids_running_sum + i

    print(f"The sum of all invalid IDs is: {invalid_ids_running_sum}")
    press_enter_to_continue()


def find_joltage_program():
    joltage_filename = "data/joltage_input.txt"
    total_output_joltage = 0

    def find_max_joltage(joltage_rating):
        str_jr = str(joltage_rating)

        possible_values = []

        for i in range(len(str_jr)):
            for j in range(i + 1, len(str_jr)):
                v = str_jr[i] + str_jr[j]
                possible_values.append(int(v))

        max_joltage = max(possible_values)
        return max_joltage

    with open(joltage_filename, "r") as file:
        for line in file:
            joltage_rating = int(line.strip())
            total_output_joltage = total_output_joltage + find_max_joltage(
                joltage_rating
            )

    print(f"The total output joltage is: {total_output_joltage}")
    press_enter_to_continue()


def find_joltage_program_2():
    joltage_filename = "data/joltage_input.txt"
    total_output_joltage = 0

    def find_max_joltage_2(joltage_rating):
        str_jr = str(joltage_rating)
        max_joltage = 0

        # This works, but is extremely slow
        # Need to find a better way to do this
        for i in itertools.combinations(list(range(len(str_jr))), 12):
            v = int("".join([str_jr[index] for index in i]))
            if v > max_joltage:
                max_joltage = v
        return max_joltage

    print("Calculating total output joltage..", end="", flush=True)
    with open(joltage_filename, "r") as file:
        for line in file:
            print(".", end="", flush=True)
            total_output_joltage = total_output_joltage + find_max_joltage_2(
                line.strip()
            )

    print(f"The total output joltage is: {total_output_joltage}")
    press_enter_to_continue()


def folklift_access_program():
    input_filename = "data/forklift_access_input.txt"

    working_floor = []
    line_length = 0
    n_lines = 0
    accessible_rolls = 0

    # Load data
    with open(input_filename, "r") as file:
        for line in file:
            line_array = list(line.strip())
            if line_length == 0:
                line_length = len(line_array)
            elif len(line_array) != line_length:
                print("Error: Inconsistent line lengths in input file.")
                return
            working_floor.append(line_array)
            n_lines += 1

    output_floor = copy.deepcopy(working_floor)

    def check_is_roll(i, j):
        if working_floor[i][j] == "@":
            return True
        return False

    def check_if_accessible(i, j):
        n_surrounding_roles = 0

        directions = [
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
        ]
        for direction in directions:
            ni, nj = i + direction[0], j + direction[1]
            if 0 <= ni < n_lines and 0 <= nj < line_length:
                if check_is_roll(ni, nj):
                    n_surrounding_roles += 1

        if n_surrounding_roles < 4:
            return True
        return False

    for i in range(n_lines):
        for j in range(line_length):
            if check_is_roll(i, j):
                if check_if_accessible(i, j):
                    output_floor[i][j] = "x"  # Mark as accessible
                    accessible_rolls += 1

    print(f"Number of accessible forklift rolls: {accessible_rolls}")

    if debug:
        for row in output_floor:
            print(" ".join(row))

    press_enter_to_continue()


def folklift_access_program_2():
    input_filename = "data/forklift_access_input.txt"

    total_removable_rolls = 0

    working_floor = []
    line_length = 0
    n_lines = 0
    accessible_rolls = 0

    # Load data
    with open(input_filename, "r") as file:
        for line in file:
            line_array = list(line.strip())
            if line_length == 0:
                line_length = len(line_array)
            elif len(line_array) != line_length:
                print("Error: Inconsistent line lengths in input file.")
                return
            working_floor.append(line_array)
            n_lines += 1

    output_floor = copy.deepcopy(working_floor)

    def check_is_roll(i, j):
        if working_floor[i][j] == "@":
            return True
        return False

    def check_if_accessible(i, j):
        n_surrounding_roles = 0

        directions = [
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
        ]
        for direction in directions:
            ni, nj = i + direction[0], j + direction[1]
            if 0 <= ni < n_lines and 0 <= nj < line_length:
                if check_is_roll(ni, nj):
                    n_surrounding_roles += 1

        if n_surrounding_roles < 4:
            return True
        return False

    rolls_removed = True
    while rolls_removed:
        rolls_removed = False
        accessible_rolls = 0

        for i in range(n_lines):
            for j in range(line_length):
                if check_is_roll(i, j):
                    if check_if_accessible(i, j):
                        output_floor[i][j] = "."  # Mark as removed in the output
                        accessible_rolls += 1
                        rolls_removed = True

        working_floor = copy.deepcopy(output_floor)

        total_removable_rolls += accessible_rolls

    print(f"Total number of removable forklift rolls: {total_removable_rolls}")

    press_enter_to_continue()


def ingredients_list_program():
    input_filename = "data/ingredients_list.txt"
    fresh_ingredients_ranges = list()
    n_fresh_ingredients = 0
    mode = "read_fresh"

    for line in open(input_filename, "r"):
        if line.strip() == "":
            mode = "read_available"
            continue
        elif mode == "read_fresh":
            [start, end] = line.strip().split("-")
            fresh_ingredients_ranges.append([int(start), int(end)])
            continue
        elif mode == "read_available":
            ingredient = int(line.strip())
            ingredient_is_fresh = False
            for r in fresh_ingredients_ranges:
                if ingredient >= r[0] and ingredient <= r[1]:
                    if debug:
                        print(f"Ingredient {ingredient} is fresh")
                    ingredient_is_fresh = True
            if ingredient_is_fresh:
                n_fresh_ingredients += 1

    print(f"Number of fresh ingredients available: {n_fresh_ingredients}")
    press_enter_to_continue()


def ingredients_list_program_2():
    from intervaltree import Interval, IntervalTree

    input_filename = "data/ingredients_list.txt"

    mode = "read_fresh"
    fresh_intervals = IntervalTree()
    for line in open(input_filename, "r"):
        if line.strip() == "":
            mode = "read_available"
            continue
        elif mode == "read_fresh":
            [start, end] = line.strip().split("-")
            fresh_intervals.add(Interval(int(start), int(end) + 1))
            continue

    # Merge overlaps
    fresh_intervals.merge_overlaps()

    total_fresh_ingredients = 0
    for interval_obj in fresh_intervals:
        total_fresh_ingredients += interval_obj.end - interval_obj.begin

    print(f"Total fresh ingredient count {total_fresh_ingredients}")
    press_enter_to_continue()


def cephalopod_math_homework_program():
    input_file_name = "data/cephalopod_math_homework.txt"

    from collections import defaultdict

    data = defaultdict(list)

    for line in open(input_file_name, "r"):
        entries = enumerate(line.strip().split())
        for index, entry in entries:
            data[index].append(entry)

    total = 0
    for index in data:
        column_data = data[index][:-1]
        column_operation = str(data[index][-1])

        op = column_operation.join(column_data)
        res = eval(op)

        total += res

    print(f"The total of all columns is: {total}")
    press_enter_to_continue()


def cephalopod_math_homework_program_2():
    print("Cephalopod Math Homework 2 is not yet implemented.")
    press_enter_to_continue()


def tachyon_manifold_program():
    mode = "initialize"

    n_splits = 0

    tachyon_positions = set()
    new_tachyon_positions = set()

    for line in open("data/tachyon_manifold.txt", "r"):
        line_array = list(line.strip())

        if debug:
            print(line_array)

        if mode == "initialize":
            tachyon_positions = [i for i, x in enumerate(line_array) if x == "S"]
            mode = "propagate"
            continue
        elif mode == "propagate":
            for i in tachyon_positions:
                if line_array[i] == "^":
                    new_tachyon_positions.add(i - 1)
                    new_tachyon_positions.add(i + 1)
                    new_tachyon_positions.remove(i)
                    n_splits += 1

                elif line_array[i] == ".":
                    new_tachyon_positions.add(i)

            tachyon_positions = copy.deepcopy(new_tachyon_positions)
            continue

    print(f"The tachyon manifold caused {n_splits} splits.")
    press_enter_to_continue()


def playground_program(
    input_data_filename="data/day8_playground_test.txt",
    max_distance=10000000.0,
    n_connections=1000,
):
    import numpy as np
    from scipy.spatial import KDTree
    import scipy.sparse as sp
    import igraph as ig

    # Load points from CSV
    points = np.genfromtxt(input_data_filename, delimiter=",")
    if debug:
        print(f"Loaded {points.shape[0]} points from {input_data_filename}")

    # Create graph and add vertices
    g = ig.Graph()
    g.add_vertices(points.shape[0])

    # Calculate distance matrix
    kd_tree = KDTree(points)
    sdm = kd_tree.sparse_distance_matrix(kd_tree, max_distance=max_distance)

    # Find the cutoff for the the top N closest edges
    distances = sp.tril(sdm, k=-1).data
    distances_sorted = np.sort(distances)
    if debug:
        print(f"distances_sorted: {distances_sorted}")
    pairs_cutoff = distances_sorted[n_connections - 1]

    # Add edges to the graph based on the sparse distance matrix
    for (u, v), dist in sdm.items():
        if dist <= pairs_cutoff:
            g.add_edge(u, v, weight=dist)
    if debug:
        print(g.summary())

    # Find the connected components and generate answers
    components = g.components()
    print(f"Number of components: {len(components)}")

    component_lengths = sorted([len(c) for c in components], reverse=True)[0:3]
    print(f"Component lengths: {component_lengths}")

    answer = eval("*".join([str(cl) for cl in component_lengths]))
    print(f"Playground answer: {answer}")

    press_enter_to_continue()


def playground_program_2(
    input_data_filename="data/day8_playground_test.txt",
    max_distance=10000000.0,
    n_connections=1000,
):
    import numpy as np
    from scipy.spatial import KDTree
    import igraph as ig

    # Load points from CSV
    points = np.genfromtxt(input_data_filename, delimiter=",")
    if debug:
        print(f"Loaded {points.shape[0]} points from {input_data_filename}")

    # Create graph and add vertices
    g = ig.Graph()
    g.add_vertices(points.shape[0])

    # Calculate distance matrix
    kd_tree = KDTree(points)
    sdm = kd_tree.sparse_distance_matrix(kd_tree, max_distance=max_distance)

    # Convert sparse distance matrix to list of edges
    u_s = []
    v_s = []
    d_s = []
    for item in sdm.items():
        if item[0][0] < item[0][1]:
            u_s.append(item[0][0])
            v_s.append(item[0][1])
            d_s.append(item[1])

    distances = np.array([u_s, v_s, d_s]).transpose()

    if debug:
        print(f"distances shape: {distances.shape}")

    sort_indices = distances[:, 2].argsort()
    sorted_distances = distances[sort_indices]

    last_row = None
    for row in sorted_distances:
        last_row = row
        u = int(row[0])
        v = int(row[1])
        d = row[2]
        g.add_edge(u, v, weight=d)
        if len(g.components()) == 1:
            # Everything is connected
            break

    if debug:
        print(f"Last added edge to connect everything: {last_row}")

    # Calculate answer
    answer = points[int(last_row[0])][0] * points[int(last_row[1])][0]
    print(f"Playground II answer: {answer}")

    press_enter_to_continue()


def red_tiles_program(filename="data/red_tiles_input.txt"):
    tile_coordinates = list()

    # Load data
    for line in open(filename, "r"):
        [x, y] = line.strip().split(",")
        tile_coordinates.append((int(x), int(y)))

    # Naive approach comparing all combinations
    max_surface_area = 0
    combinations_of_two = itertools.combinations(tile_coordinates, 2)

    for (x1, y1), (x2, y2) in combinations_of_two:
        surface_area = abs(x1 - x2 + 1) * abs(y1 - y2 + 1)
        if surface_area > max_surface_area:
            max_surface_area = surface_area

    print(f"The maximum surface area of red tiles is: {max_surface_area}")

    press_enter_to_continue()


def factory_program(input_file="data/day10_factory_data.txt"):
    import igraph as ig

    def state_string_to_int(state_string):
        digits_text = list(re.findall(r"[\.#]+", state_string)[0])
        map_dict = {".": "0", "#": "1"}
        state_in_binary = "".join([map_dict[item] for item in digits_text])
        state = int("".join(reversed(state_in_binary)), 2)

        return state

    def get_max_state(state_string):
        digits_text = list(re.findall(r"[\.#]+", state_string)[0])
        return pow(2, len(digits_text))

    def apply_transition_to_state(state: int, transition: list[int]):
        pass

    def tuple_string_to_bits(s: str) -> int:
        # Remove parentheses and whitespace
        s = s.strip()[1:-1]
        bits = 0
        for n in s.split(","):
            bits |= 1 << int(n)
        return bits

    def process_factory_line(line):
        # Parse line with regex
        capture_pattern = r"^(\[.*\]) (\(.*\)) (\{.*\})$"
        target_state_string, button_presses_string, joltage_string = re.findall(
            capture_pattern, line
        )[0]

        target_state_int = state_string_to_int(target_state_string)
        max_state = get_max_state(target_state_string)

        if debug:
            print(f"target_state_string: {target_state_string}")
            print(f"button_presses_string: {button_presses_string}")
            print(f"joltage_string: {joltage_string}")
            print("---")
            print(f"target_state_int: {target_state_int}")
            print(f"max_state: {max_state}")
            print("---")

        # Parse transitions
        transitions = [
            tuple_string_to_bits(x) for x in button_presses_string.split(" ")
        ]

        if debug:
            print("Transitions: ")
            print(transitions)

        # Generate igraph object with states
        g = ig.Graph()
        g.add_vertices(max_state)

        # For each state
        for v in g.vs:
            cur_index = v.index
            # For each button
            for t in transitions:
                to_index = cur_index ^ t  # bitwise xor
                # Add transition
                if not g.are_adjacent(cur_index, to_index):
                    g.add_edge(cur_index, to_index)

        # Use igraph g.get_shortest_paths() to find shortest path from initial to target state
        n_button_presses = len(g.get_shortest_paths(0, target_state_int)[0]) - 1
        return n_button_presses

    total_button_presses = 0
    for line in open(input_file, "r"):
        line = line.strip()
        total_button_presses += process_factory_line(line)

    print(f"The sum of the minimum number of button presses is: {total_button_presses}")

    press_enter_to_continue()


def factory_program_2(input_file="data/day10_factory_data.txt"):
    from scipy.optimize import linprog
    import numpy as np

    def process_factory_line(line):
        # Parse line with regex
        capture_pattern = r"^(\[.*\]) (\(.*\)) (\{.*\})$"
        target_state_string, button_presses_string, joltage_string = re.findall(
            capture_pattern, line
        )[0]

        # Space dimentionality for this specific machine
        space_dimentionality = len(re.findall(r"[\.#]+", target_state_string)[0])

        # One hot encode transitions
        def encode_transitions(s: str, size: int):
            result = []

            # find contents inside parentheses
            groups = re.findall(r"\(([^)]*)\)", s)

            for group in groups:
                values = [int(x.strip()) for x in group.split(",")]
                encoded = [0] * size
                for v in values:
                    encoded[v] = 1
                result.append(encoded)

            return result

        # Get states
        target_state = [int(y) for y in joltage_string[1:-1].split(",")]
        allowed_transitions = encode_transitions(
            button_presses_string, space_dimentionality
        )

        # This is now a linear programming problem
        c = np.ones([len(allowed_transitions)])  # all buttons count as 1 keypress
        A_eq = np.array(allowed_transitions).T  # vectors to combine
        b_eq = np.array(target_state)  # target equality value

        # Solve
        ans = linprog(c=c, A_eq=A_eq, b_eq=b_eq, integrality=1)
        min_button_presses = np.sum(np.round(ans.x))

        return min_button_presses

    total_min_button_presses = 0
    for line in open(input_file, "r"):
        line = line.strip()
        total_min_button_presses += process_factory_line(line)

    print(f"Total minimum number of key presses: {total_min_button_presses}")

    press_enter_to_continue()


def main_menu():
    from consolemenu import ConsoleMenu
    from consolemenu.items import FunctionItem

    menu = ConsoleMenu("Advent of Code 2025 Challenges")
    function_item_1 = FunctionItem("Day 1: Elves Dial - Part I", elves_dial_program)

    menu.append_item(function_item_1)
    function_item_2 = FunctionItem(
        "Day 1: Elves Dial 0x434C49434B - Part II", elves_dial_program_2
    )
    menu.append_item(function_item_2)
    function_item_3 = FunctionItem("Day 2: Invalid Ids - Part I", invalid_ids_program)

    menu.append_item(function_item_3)
    function_item_4 = FunctionItem(
        "Day 2: Invalid Ids - Part II", invalid_ids_program_2
    )
    menu.append_item(function_item_4)
    function_item_5 = FunctionItem(
        "Day 3: Find Joltage - Part I)", find_joltage_program
    )

    menu.append_item(function_item_5)
    function_item_6 = FunctionItem(
        "Day 3: Find Joltage II (very slow) part II", find_joltage_program_2
    )
    menu.append_item(function_item_6)
    function_item_7 = FunctionItem(
        "Day 4: Forklift Access  - part I", folklift_access_program
    )
    menu.append_item(function_item_7)
    function_item_8 = FunctionItem(
        "Day 4: Forklift Access 2 - part II", folklift_access_program_2
    )
    menu.append_item(function_item_8)
    function_item_9 = FunctionItem(
        "Day 5: Ingredients List - part I", ingredients_list_program
    )
    menu.append_item(function_item_9)
    function_item_10 = FunctionItem(
        "Day 5: Ingredients List 2 - Part II", ingredients_list_program_2
    )
    menu.append_item(function_item_10)
    function_item_11 = FunctionItem(
        "Day 6: Cephalopod Math Homework - part I", cephalopod_math_homework_program
    )
    menu.append_item(function_item_11)
    function_item_12 = FunctionItem(
        "Day 6: Cephalopod Math Homework - part II - Not working",
        cephalopod_math_homework_program_2,
    )
    menu.append_item(function_item_12)
    function_item_13 = FunctionItem(
        "Day 7: Tachyon Manifold - part I", tachyon_manifold_program
    )
    menu.append_item(function_item_13)

    function_item_14 = FunctionItem(
        "Day 7: Tachyon Manifold 2 - part II - NOT WORKING", None
    )
    menu.append_item(function_item_14)

    function_item_15 = FunctionItem("Day 8: Playground - part I", playground_program)
    menu.append_item(function_item_15)

    function_item_16 = FunctionItem("Day 8: Playground - part II", playground_program_2)
    menu.append_item(function_item_16)

    function_item_17 = FunctionItem("Day 9: Red Tiles - part I", red_tiles_program)
    menu.append_item(function_item_17)

    function_item_18 = FunctionItem("Day 9: Red Tiles - part II - NOT WORKING", None)
    menu.append_item(function_item_18)

    function_item_19 = FunctionItem("Day 10: Factory - Part I", factory_program)
    menu.append_item(function_item_19)

    function_item_20 = FunctionItem("Day 10: Factory - Part II", factory_program_2)
    menu.append_item(function_item_20)

    menu.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(prog="Advent of Code 2025 Challenges")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Set global debug flag
    if args.debug:
        global debug
        debug = True

    # Show main menu
    main_menu()
    sys.exit(0)


if __name__ == "__main__":
    main()
