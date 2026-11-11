import math

def split_detected_string(detected_string):
    numbers = []
    characters = []
    # 0 -liczba, 1 - znak, 2 - nawias, 3 - =, 4 - char np.e,x,y
    equation = []
    detected_arr = detected_string.split()
    number = 0
    for x in range(len(detected_arr)):
        if detected_arr[x].isdecimal():
            if x!=0 and detected_arr[x-1].isdecimal():
                number = number *10 + int(detected_arr[x])
            else:
                number = int(detected_arr[x])
        else:
            if number !=0:
                numbers.append(number)
                equation.append(0)
                number = 0
            if detected_arr[x] == "plus" or detected_arr[x] == "minus" or detected_arr[x] == "mnozenie_gwiazdka" or detected_arr[x] == "mnozenie_kropka" or detected_arr[x] == "mnozenie_x" or detected_arr[x] =="dzielenie_dwukropek" or detected_arr[x] =="dzielenie_slash" or detected_arr[x] =="dzielenie_ulamek" or detected_arr[x] == "modulo" or detected_arr[x] =="calka" or detected_arr[x] =="pochodna" or detected_arr[x] == "symbol_sumy" or detected_arr[x] =="znak_iloczynu" or detected_arr[x] =="pierwiastek" or detected_arr[x] == "silnia":
                equation.append(1)
                characters.append(detected_arr[x])
            elif detected_arr[x] == "otwarcie1" or detected_arr[x] == "zamkniecie1" or detected_arr[x] == "otwarcie2" or detected_arr[x] == "zamkniecie2" or detected_arr[x] == "otwarcie3" or detected_arr[x] == "zamkniecie3":
                equation.append(2)
            elif detected_arr[x] == "rowna_sie":
                equation.append(3)
            elif detected_arr[x] == "e" or detected_arr[x] =="pi" or detected_arr[x] =="x" or detected_arr[x] == "y" or detected_arr[x] == "z":
                equation.append(4)
                characters.append(detected_arr[x])
    return equation, characters, numbers

def middle_symbol(num1, num2, character):
    match character:
        case "plus":
            return num1 + num2
        case "minus":
            return num1 - num2
        case "mnozenie_gwiazdka":
            return num1 * num2
        case "mnozenie_kropka":
            return num1 * num2
        case "mnozenie_x":
            return num1 * num2
        case "dzielenie_dwukropek":
            return num1 / num2
        case "dzielenie_slash":
            return num1 / num2
        case "dzielenie_ulamek":
            return num1 / num2
        case "modulo":
            return num1 % num2

def symbol_first(num, character):
    match character:
        case "całka":
            return 0
        case "pochodna":
            return 1
        case "symbol_sumy":
            return 2
        case "znak_iloczynu":
            return 3
        case "pierwiastek":
            return 4
#     if character = "silnia"

def character(num, char):
    match char:
        case "e":
            return num*math.e
        case "pi":
            return num*math.pi
        case "x":
            return 0
        case "y":
            return 0
        case "z":
            return 0
def bracket(num, char):
    match char:
        case "plus":
            return 0

def equation(equation, character, number):
    for x in range(len(equation)):
        match equation[x]:
            case 1:
                if x != 0 and equation[x-1] == 0 or equation[x-1] == 4 or equation[x-1] == 2:
                    middle_symbol(equation[x-1], character)



