"""
    Это документационная строка для main.py.

    Описание алгоритма.
Получаем кадры с обеих камер
1) Изначально, если айди объекта сменился на новый, увеличиваем счетчик кадров, проверяем наличие массива array_id,
	если под прошлым айди был найден уже существующий объект,
		то увеличиваем количество одинаковых объектов
		очищаем элемент массива векторов относящийся к этому объекту
Если объектов во втором кадре нет, то так же проверяем array_id со второй камеры, чтобы увеличить счетчик одинаковых элементов

Далее проверка нового объекта
Если счетчик кадров больше 40, считаем вектор текущего элемента как 0.9 старого значения и 0.1 нового и заносим в массив векторов под текущим id
	Параллельно считаем сходство данного вектора с уже имеющимися векторами, перебирая массив векторов.
		Если значение параметра схожести векторов больше нужной величины, то в массиве array_id инкрементируем элемент под найденным id
Иначе ищем максимальный элемент массива array_id,
	если он больше хотя бы 1/6 количества кадров определения айди, то записываем в correct_id номер элемента
	иначе отдаем ему id идущий по порядку

2) Аналогично для второй камеры

***Трекер работает по принципу подсчета расстояния между центрами поступающих ему боксов
***Перед входом в циклы, есть замер времени и поиск центра бокса, это нужно для трекера,
если сеть на несколько кадров теряет человека, то если время и расстояние между центрами последнего бокса и нового меньше определенной величины, то отдаем ему тот же айди

    Пример использования:
    python main.py 3.Camera 4.Camera GeneralNMHuman_v1.0_IR10_FP16 original_reid
    """


import argparse
import cmath
from datetime import datetime

import cv2
import numpy as np

from network import NeuralNetworkDetector
from small_utils import (
    calculate_area,
    center_point_save,
    clean_array,
    get_screen_resolution,
    if_border,
    resize_frame,
)
from tracker import EuclideanDistTracker

parser = argparse.ArgumentParser(description="Input video file")
parser.add_argument("video_path1", type=str, help="Path to the video file number 1")
parser.add_argument("video_path2", type=str, help="Path to the video file number 2")
parser.add_argument("openvino_path", type=str, help="Path to the openvino model")
parser.add_argument("onnx_path", type=str, help="Path to the onnx model")
args = parser.parse_args()

EPSILON = 1e-6


# метод детекции людей с помощью нейросети
# Input:
# frame - кадр с входного видео
# Output:
# координаты найденных боксов;
def net_find_box(frame, detections, size):
    input_frame = detector_openvino.data_preparation(frame)
    outs_net = detector_openvino.forward(input_frame)  # результат работы нейронной сети
    outs_net = outs_net[0][0]
    for out in outs_net:
        coord_array = []
        if out[2] == 0.0:
            break
        if out[2] > size_out:
            coord_array.append(
                [out[3], out[4], out[5], out[6]]
            )  # добавление в массив координат

            coord_full_size = coord_array[0]
            h = frame.shape[0]
            w = frame.shape[1]
            coord_full_size = coord_full_size * np.array(
                [w, h, w, h]
            )  # преобразование относительных координат в абсолютные на исходном изображении
            coord_full_size = coord_full_size.astype(
                np.int32
            )  # возвращение копии массива, преобразованного к указанному типу (вещественные числа с одинарной точностью)
            area = calculate_area(
                coord_full_size[0],
                coord_full_size[1],
                coord_full_size[2],
                coord_full_size[3],
            )
            if area > size[0] and area < size[1]:
                detections.append(
                    [
                        coord_full_size[0],
                        coord_full_size[1],
                        coord_full_size[2],
                        coord_full_size[3],
                    ]
                )  # добавление найденных координат
        return


# метод поиска вектора с помощью нейросети
# Input:
# frame - бокс с найденным человеком
# Output:
# вектор
def net_search_vector(frame, global_id):
    alpha = 0.9
    input_frame = detector_onnx.data_preparation(frame)
    outs_vector = detector_onnx.forward(input_frame)
    if np.all(np.abs(vector[global_id]) < EPSILON):
        vector[global_id] = outs_vector
    else:
        vector[global_id] = alpha * vector[global_id] + (1 - alpha) * outs_vector
    return


# метод на случай, если найден тот же объект
# если максимальное число совпадений больше, чем хотя бы 1/6 от количества кадров проверки,
# то очищаем текущий элемент с собранным вектором для этого объекта, так как он уже есть
# + увеличиваем количество совпадающих объектов
# Input:
# arr_id - массив с совпадающими айди, count_same - подсчет одинаковых элементов, id_save - айди предыдущего объекта
# Output:
# количество одинаковых элементов
def if_same_object(arr_id, count_same, id_save):
    # если был найден существующий объект, то очистка собранного вектора и массива id + увеличение числа совпадающих объектов
    if max(arr_id) >= count_frame // 6:
        vector[id_save - count_same] = 0
        count_same += 1
    clean_array(arr_id)
    return count_same


# метод для определения корректного айди для объекта
# если максимальное число совпадений больше, чем хотя бы 1/6 от количества кадров проверки,
# то отдаем объекту найденный айди,
# иначе айди по порядку
# Input:
# arr_id - массив с совпадающими айди, count_same - подсчет одинаковых элементов, global_id - айди текущего объекта
# Output:
# корректный айди объекта
def find_max_same_id(arr_id, count_same, global_id):
    arr_id_list = arr_id.tolist()
    # если максимальное совпадение больше чем 1/6 от общего числа кадров проверки
    if max(arr_id_list) >= count_frame // 6:
        # то берем найденный id
        corr_id = arr_id_list.index(max(arr_id_list))
    else:
        # иначе следующий по порядку
        corr_id = global_id - count_same
    return corr_id


# метод поиска схожести векторов с помощью нейросети
# Input:
# vector1, vector2 - векторы объектов
# Output:
# величина отражающая схожесть двух векторов
def net_search_compare(vector1, vector2):
    mult, sqra, sqrb = (0, 0, 0)
    for i in range(vector_size):
        a = vector1[i]
        b = vector2[i]
        mult += a * b
        sqra += a * a
        sqrb += b * b
    return mult / (cmath.sqrt(sqra * sqrb))


# метод подсчета совпадений
# Input:
# global_id - id текущего объекта по порядку, opt_param - наименьший параметр схожести,
# arr_id - массив, содержащий количество совпадений с каждым существующим id
# Output:
# заполненный массив arr_id
def net_count_compare(global_id, opt_param, arr_id):
    max_c = 0
    obj = -1
    for i in range(global_id):
        compare = net_search_compare(vector[global_id], vector[i])
        if compare > opt_param:
            if max_c < compare:
                max_c = compare
                obj = i
    if obj >= 0:
        # увеличиваем значение элемента с индексом id у которого максимальное совпадение с вектором
        arr_id[obj] += 1
    return


# метод для увеличения count_same - если новый объект, а старый был уже существующим
# Input:
# id текущего объекта по порядку, num_cam, num_cam2 - наименьший параметр схожести,
# Output:
# заполненный массив arr_id
def if_new_object(id, num_cam, num_cam2):
    global count_same
    # если новый объект
    if id != id_save[num_cam]:
        # проверяем содержимое массива id для первой камеры и увеличиваем кол-во одинаковых объектов
        count_same = if_same_object(array_id[num_cam], count_same, id_save[num_cam])
        # если нет объекта во второй камере
        if report[num_cam2]:
            # проверяем содержимое массива id для второй камеры и увеличиваем кол-во одинаковых объектов
            count_same = if_same_object(
                array_id[num_cam2], count_same, id_save[num_cam2]
            )
        # обновляем счетчик кадров для гистограмм и для детектора
        count_frame_cam[num_cam] = count_frame


# метод для определения нового объекта на полученном кадре
# Input:
# boxes_ids - координаты бокса и id нового объекта, num_cam - номер камеры, frame - сам кадр
# Output:
# верно определенный id объекта и бокс на кадре
def camera_tracking(boxes_ids, num_cam, frame):
    global count_same, correct_id
    num_cam2 = (num_cam + 1) % 2
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        # если надпись окажется за пределами
        y1 = if_border(y, h)
        report[num_cam] = False
        frame_plt1 = frame[y + 10 : h, x:w]
        # если новый объект, проверяем наличие массивов id (второй массив, только если отсутствует объект во 2 кадре)
        if_new_object(id, num_cam, num_cam2)
        global_id1 = id - count_same
        id_save[num_cam] = id
        # пока счетчик кадров больше нуля, считаем сходства новой гистограммы с уже имеющимися
        if count_frame_cam[num_cam] > 0:
            # ищем вектор схожести
            net_search_vector(frame_plt1, global_id1)
            # считаем сходства с существующими id
            net_count_compare(global_id1, opt_param[num_cam], array_id[num_cam])
            count_frame_cam[num_cam] -= 1
            correct_id[num_cam] = global_id1
        else:
            # иначе ищем id с макс совпадением
            correct_id[num_cam] = find_max_same_id(array_id[num_cam], count_same, id)
            id = correct_id[num_cam]
            cv2.putText(
                frame,
                "Object " + str(id),
                (x, y1),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )
        center[num_cam] = center_point_save(x, w, y, h)
        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 3)
        time[num_cam][0] = datetime.now().timestamp()
        tracker[num_cam2].id_count = tracker[num_cam].id_count
    return


# метод для обновления трекера и использование метода camera_tracking
# Input:
# frame - сам кадр, num_cam - номер камеры, size - минимальный и максимальный размер бокса
# Output:
# output метода camera_tracking
def update_camera_tracking(frame, num_cam, size):
    detections = []
    net_find_box(frame, detections, size)
    time[num_cam][1] = datetime.now().timestamp()
    # разница времени с последнего бокса в первом кадре с текущим моментом
    delta_time = int(time[num_cam][1]) - int(time[num_cam][0])
    report[num_cam] = True
    # обновление трекера
    boxes_ids1 = tracker[num_cam].update(detections, center[num_cam], delta_time)
    # трекинг, назначение id, отрисовка боксов
    camera_tracking(boxes_ids1, num_cam, frame)
    return


# Путь к файлам модели и весам
model_path_openvino = args.openvino_path
model_path_onnx = args.onnx_path
# размеры данных на входы сетей
size_openvino = (300, 300)
size_onnx = (128, 256)
# загрузка нейронной сети для детекции из файлов и выбор устройства для инференса
detector_openvino = NeuralNetworkDetector(
    model_format="openvino", model_path=model_path_openvino, size=size_openvino
)
# загрузка нейронной сети для поиска схожести из файлов и выбор устройства для инференса
detector_onnx = NeuralNetworkDetector(
    model_format="onnx", model_path=model_path_onnx, size=size_onnx
)

size_out = 0.3  # минимальная уверенность сети

tracker = [EuclideanDistTracker(), EuclideanDistTracker()]

video_path1 = args.video_path1 + ".avi"
video_path2 = args.video_path2 + ".avi"
cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)
width, height = get_screen_resolution()

id_save = [-1, -1]
count_frame = 40  # количество кадров для идентификации человека
vector_size = 256  # размер выходного вектора из сети onnx

count_frame_cam = [0, 0]
ret1, frame1_1 = cap1.read()
ret2, frame2_2 = cap2.read()
height_frame, width_frame, _ = frame2_2.shape

# минимальные и максимальные размеры боксов
size_box1 = (
    int(round(0.045 * width_frame * height_frame)),
    int(round(width_frame * height_frame / 3)),
)
size_box2 = (
    int(round(0.0076 * width_frame * height_frame)),
    int(round(width_frame * height_frame / 3)),
)

array_id = np.zeros((2, 10))  # массив для накопления совпадений с конкретным объектом

vector = np.zeros((200, vector_size))
opt_param = [0.65, 0.48]  # границы сравнения векторов для каждого кадра
count_same = 0  # переменная для подсчета одинаковых объектов

report = [True, True]
center = np.zeros((2, 2))
correct_id = [0, 0]
time = np.zeros((2, 2))


def main():
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # трекинг кадра с первой камеры
        update_camera_tracking(frame1, 0, size_box1)

        # трекинг кадра со второй камеры
        update_camera_tracking(frame2, 1, size_box2)

        combined_frame = resize_frame(frame1, frame2, width, height)
        cv2.imshow("combined", combined_frame)
        key = cv2.waitKey(30)
        if key == 27:
            break
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
