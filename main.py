import functools
import sys
import re
import itertools
import copy
import gc

debug = False

@functools.lru_cache(maxsize=None)
def fibonnaci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonnaci(n - 1) + fibonnaci(n - 2)
    
def elves_dial_program():
    start_position = 50
    n_times_pointing_zero = 0

    filename = 'elves_rotations.txt'
    with open(filename, 'r') as file:
        for line in file:
            rotation = line.strip()
            direction = rotation[0]
            steps = int(rotation[1:])

            if direction == 'L':
                start_position = (start_position - steps) % 100
            elif direction == 'R':
                start_position = (start_position + steps) % 100
            else:
                print(f"Invalid direction '{direction}' in line: {line}")
                continue
            
            if start_position == 0:
                n_times_pointing_zero += 1

    print(f"The dial pointed to 0 a total of {n_times_pointing_zero} times.")



def elves_dial_program_2():
    start_position = 50
    n_times_pointing_zero = 0

    filename = 'elves_rotations.txt'
    with open(filename, 'r') as file:
        for line in file:
            rotation = line.strip()
            direction = rotation[0]
            steps = int(rotation[1:])

            if direction == 'L':
                while steps > 0:
                    start_position = (start_position - 1) % 100
                    steps -= 1
                    if start_position == 0:
                        n_times_pointing_zero += 1
            elif direction == 'R':
                while steps > 0:
                    start_position = (start_position + 1) % 100
                    steps -= 1
                    if start_position == 0:
                        n_times_pointing_zero += 1

            else:
                print(f"Invalid direction '{direction}' in line: {line}")
                continue
            
    print(f"The dial pointed to 0 a total of {n_times_pointing_zero} times using method 0x434C49434B.")

def invalid_ids_program():
    invalid_ids_running_sum = 0
    invalid_ids_filename = 'invalid_ids.txt'

    def is_invalid_id(id):
        str_id = str(id)
        pattern = r"^(.*?)\1$"
        match = re.search(pattern, str_id)
        if match:
            return True

        return False

    with open(invalid_ids_filename, 'r') as file:
        line = file.readline()
        ranges_list = line.split(',')
        for r in ranges_list:
            start, end = map(int, r.split('-'))
            for i in range(start, end + 1):
                if(is_invalid_id(i)):
                    invalid_ids_running_sum = invalid_ids_running_sum + i

    print(f"The sum of all invalid IDs is: {invalid_ids_running_sum}")


def invalid_ids_program_2():
    invalid_ids_running_sum = 0
    invalid_ids_filename = 'invalid_ids.txt'

    def is_invalid_id(id):
        str_id = str(id)
        pattern = r"^([0-9]{1,})\1{1,}$"
        match = re.search(pattern, str_id)
        if match:
            if debug:
                print(f"Invalid ID found: {str_id}")
            return True

        return False

    with open(invalid_ids_filename, 'r') as file:
        line = file.readline()
        ranges_list = line.split(',')
        for r in ranges_list:
            start, end = map(int, r.split('-'))
            for i in range(start, end + 1):
                if(is_invalid_id(i)):
                    invalid_ids_running_sum = invalid_ids_running_sum + i

    print(f"The sum of all invalid IDs is: {invalid_ids_running_sum}")


def find_joltage_program():
    joltage_filename = 'joltage_input.txt'
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

    with open(joltage_filename, 'r') as file:
        for line in file:
            joltage_rating = int(line.strip())
            total_output_joltage = total_output_joltage + find_max_joltage(joltage_rating)

    print(f"The total output joltage is: {total_output_joltage}")




def find_joltage_program_2():
    joltage_filename = 'joltage_input.txt'
    total_output_joltage = 0

    def find_max_joltage_2(joltage_rating):
        str_jr = str(joltage_rating)
        max_joltage = 0

        # This works, but is extremely slow
        # Need to find a better way to do this
        for i in itertools.combinations(list(range(len(str_jr))),12):
            v = int(''.join([str_jr[index] for index in i]))
            if v > max_joltage:
                max_joltage = v
        return max_joltage

    print("Calculating total output joltage..", end='', flush=True)
    with open(joltage_filename, 'r') as file:
        for line in file:
            print(".", end='', flush=True)
            total_output_joltage = total_output_joltage + find_max_joltage_2(line.strip())

    print(f"The total output joltage is: {total_output_joltage}")


def folklift_access_program():
    input_filename = 'forklift_access_input.txt'

    working_floor = []
    line_length = 0
    n_lines = 0
    accessible_rolls = 0

    # Load data
    with open(input_filename, 'r') as file:
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
        if working_floor[i][j] == '@':
            return True
        return False    

    def check_if_accessible(i, j):
        n_surrounding_roles = 0

        directions = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
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
            if check_is_roll(i,j):
                if check_if_accessible(i, j):
                    output_floor[i][j] = 'x'  # Mark as accessible
                    accessible_rolls += 1

    print(f"Number of accessible forklift rolls: {accessible_rolls}")

    if debug:
        for row in output_floor:
            print(' '.join(row))


def folklift_access_program_2():
    input_filename = 'forklift_access_input.txt'

    total_removable_rolls = 0

    working_floor = []
    line_length = 0
    n_lines = 0
    accessible_rolls = 0

    # Load data
    with open(input_filename, 'r') as file:
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
        if working_floor[i][j] == '@':
            return True
        return False    

    def check_if_accessible(i, j):
        n_surrounding_roles = 0

        directions = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
        for direction in directions:
            ni, nj = i + direction[0], j + direction[1]
            if 0 <= ni < n_lines and 0 <= nj < line_length:
                if check_is_roll(ni, nj):
                    n_surrounding_roles += 1

        if n_surrounding_roles < 4:
            return True
        return False

    rolls_removed = True
    while rolls_removed == True:
        rolls_removed = False
        accessible_rolls = 0

        for i in range(n_lines):
            for j in range(line_length):
                if check_is_roll(i,j):
                    if check_if_accessible(i, j):
                        output_floor[i][j] = '.'  # Mark as removed in the output
                        accessible_rolls += 1
                        rolls_removed = True

        working_floor = copy.deepcopy(output_floor)

        total_removable_rolls += accessible_rolls

    print(f"Total number of removable forklift rolls: {total_removable_rolls}")


def ingredients_list_program():
    input_filename = 'ingredients_list.txt'
    fresh_ingredients_ranges = list()
    n_fresh_ingredients = 0
    mode = 'read_fresh'


    for line in open(input_filename, 'r'):
        if line.strip() == '':
            mode = 'read_available'
            continue
        elif mode == 'read_fresh':
            [start, end] = line.strip().split('-')
            fresh_ingredients_ranges.append([int(start), int(end)])
            continue
        elif mode == 'read_available':
            ingredient = int(line.strip())
            ingredient_is_fresh = False
            for r in fresh_ingredients_ranges:
                if ingredient >= r[0] and ingredient <= r[1]:
                    print(f'Ingredient {ingredient} is fresh')
                    ingredient_is_fresh = True
            if ingredient_is_fresh:
                n_fresh_ingredients += 1

    print(f"Number of fresh ingredients available: {n_fresh_ingredients}")

def ingredients_list_program_2():
    from intervaltree import Interval, IntervalTree

    input_filename = 'ingredients_list.txt'

    mode = 'read_fresh'
    fresh_intervals = IntervalTree()
    for line in open(input_filename, 'r'):
        if line.strip() == '':
            mode = 'read_available'
            continue
        elif mode == 'read_fresh':
            [start, end] = line.strip().split('-')
            fresh_intervals.add(Interval(int(start),int(end)+1))
            continue

    # Merge overlaps
    fresh_intervals.merge_overlaps()

    total_fresh_ingredients = 0
    for interval_obj in fresh_intervals:
        total_fresh_ingredients += interval_obj.end - interval_obj.begin

    print(f"Total fresh ingredient count {total_fresh_ingredients}")

def cephalopod_math_homework_program():
    input_file_name = 'cephalopod_math_homework.txt'

    from collections import defaultdict

    data = defaultdict(list)

    for line in open(input_file_name, 'r'):
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


def cephalopod_math_homework_program_2():
    print("Cephalopod Math Homework 2 is not yet implemented.")
        

def tachyon_manifold_program():

    mode = 'initialize'

    n_splits = 0

    tachyon_positions = set()
    new_tachyon_positions = set()

    for line in open('tachyon_manifold.txt', 'r'):
        line_array = list(line.strip())

        print(line_array)

        if mode == 'initialize':
            tachyon_positions = [i for i, x in enumerate(line_array) if x == 'S']
            mode = 'propagate'
            continue
        elif mode == 'propagate':
            
            for i in tachyon_positions:
                if line_array[i] == '^':
                    new_tachyon_positions.add(i - 1)
                    new_tachyon_positions.add(i + 1)
                    new_tachyon_positions.remove(i)
                    n_splits += 1

                elif line_array[i] == '.':
                    new_tachyon_positions.add(i)

            tachyon_positions = copy.deepcopy(new_tachyon_positions)
            continue
    
    print(f"The tachyon manifold caused {n_splits} splits.")


    

def menu():
    print("1. Calculate Fibonacci")
    print("2. Elves Dial")
    print("3. Elves Dial 0x434C49434B")
    print("4. Invalid Ids")
    print("5. Invalid Ids II")
    print("6. Find Joltage")
    print("7. Find Joltage II (very slow)")
    print("8. Forklift Access")
    print("9. Forklift Access 2")
    print("10. Ingredients List")
    print("11. Ingredients List 2")
    print("12. Cephalopod Math Homework")
    print("13. Cephalopod Math Homework (not implemented)")
    print("14. Tachyon Manifold")


    print("100. Exit")

def handle_input(choice):
    if choice == '1':
        fibonnaci_program()
    elif choice == '2':
        elves_dial_program()
    elif choice == '3':
        elves_dial_program_2()
    elif choice == '4':
        invalid_ids_program()
    elif choice == '5':
        invalid_ids_program_2()
    elif choice == '6':
        find_joltage_program()
    elif choice == '7':
        find_joltage_program_2()
    elif choice == '8':
        folklift_access_program()
    elif choice == '9':
        folklift_access_program_2()
    elif choice == '10':
        ingredients_list_program()
    elif choice == '11':
        ingredients_list_program_2()
    elif choice == '12':
        cephalopod_math_homework_program()
    elif choice == '13':
        cephalopod_math_homework_program_2()

    elif choice == '14':
        tachyon_manifold_program()

    elif choice == '100':
        print("Exiting the program.")
        sys.exit(0)
    else:
        print("Invalid choice. Please try again.")


def  fibonnaci_program():
    num = int(input("Enter a number: "))
    result = fibonnaci(num)
    print(f"The Fibonacci of {num} is {result}")

def main():
    while(True):
        menu()
        menu_option = input("Choose an option: ")
        handle_input(menu_option)



if __name__ == "__main__":
    main()