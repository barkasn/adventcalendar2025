import functools
import sys
import re

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


def menu():
    print("1. Calculate Fibonacci")
    print("2. Elves Dial")
    print("3. Elves Dial 0x434C49434B")
    print("4. Invalid Ids")
    print("5. Invalid Ids II")
    print("6. Exit")

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