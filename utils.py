import pyautogui

# метод подсчета площади бокса
# Input
# x1, y1, x2, y2 - координаты бокса
# Output:
# Площадь бокса
def calculate_area(x1, y1, x2, y2):
    return (x2 - x1)*(y2 - y1)


# метод коррекции отображения id, если бокс касается верхней границы, то запись отображается внизу бокса, иначе сверху
# Input
# y, h - координата y левого верхнего угла и высота бокса
# Output:
# скорректированная координата y
def if_border(y, h):
    return y + h + 30 if y < 50 else y - 15


# метод определения размера экрана
# Output:
# ширина и высота экрана
def get_screen_resolution():
    screen_width, screen_height = pyautogui.size()
    return screen_width, screen_height


# метод нахождения центра
# Input:
# координаты бокса
# Output:
# координаты центра
def center_point_save(x1, x2, y1, y2):
    return (x1 + x1 + x2) // 2, (y1 + y1 + y2) // 2


def clean_array(arr_id):
    for i in range(len(arr_id)):
        arr_id[i] = 0
