# Описание алгоритма:
# Получаем кадры с обеих камер
# 1) Изначально, если айди объекта сменился на новый, увеличиваем счетчик кадров, проверяем наличие массива arr_id,
# 	если под прошлым айди был найден уже существующий объект,
# 		то увеличиваем количество одинаковых объектов
# 		очищаем элемент массива векторов относящийся к этому объекту
# Если объектов во втором кадре нет, то так же провермяем arr_id со второй камеры, чтобы увеличить счетчик одинаковых элементов
#
# Далее проверка нового объекта
# Если счетчик кадров больше 40, считаем вектор текущего элемента как 0.9 старого значения и 0.1 нового и заносим в массив векторов под текущим id
# 	Параллельно считаем сходство данного вектора с уже имеющимися векторами, перебирая массив векторов.
# 		Если значение параметра схожести векторов больше нужной величины, то в массиве arr_id инкрементируем элемент под найденным id
# Иначе ищем максимальный элемент массива arr_id,
# 	если он больше хотя бы 1/6 количества кадров определения айди, то записываем в correct_id номер элемента
# 	иначе отдаем ему id идущий по порядку
#
# 2) Аналогично для второй камеры
#
# ***Трекер работает по принципу подсчета расстояния между центрами поступающих ему боксов
# ***Перед входом в циклы, есть замер времени и поиск центра бокса, это нужно для трекера,
# если сеть на несколько кадров теряет человека, то если время и расстояние между центрами последнего бокса и нового меньше определенной величины, то отдаем ему тот же айди.
import argparse
import cmath
from datetime import datetime

import cv2
import numpy as np
from openvino.inference_engine import IECore

from tracker import *
from utils import *


# класс инициализации нейронной сети
# Input:
# model_format - формат модели сети может быть onnx либо openvino, model_path - имя нейронной сети без расширения,
# size - размер входного изображения для сети, device - устройство, по умолчанию GPU
# метод forward
# Input:
# frame - изображение на вход сети
# Output:
# результат работы сети
class NeuralNetworkDetector:
    def __init__(self, model_format, model_path, size, device="GPU"):
        self.ie = IECore()
        if model_format == "openvino":
            self.net = self.ie.read_network(
                model=model_path + ".xml", weights=model_path + ".bin"
            )
        elif model_format == "onnx":
            self.net = self.ie.read_network(model=model_path + ".onnx")

        self.exec_net = self.ie.load_network(network=self.net, device_name=device)
        self.input_size = size

    def forward(self, frame):
        resized_frame = cv2.resize(
            frame, self.input_size, interpolation=cv2.INTER_AREA
        )  ## изменение изображения до размера входа сети
        input_frame = resized_frame.transpose(2, 0, 1)  # изменение RGB -> BGR
        input_name = next(
            iter(self.exec_net.input_info)
        )  # функция синхронного исполнения нейронной сети (блокирует пользовательское приложение на время выполнения запроса на вывод)
        output = self.exec_net.infer(
            {input_name: input_frame}
        )  # инференс нейронной сети
        outs_net = next(iter(output.values()))  # выходные данные нейронной сети
        return outs_net


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
    # если максимальное совпадение больше чем 1/6 от общего числа кадров проверки
    if max(arr_id) >= count_frame // 6:
        # то берем найденный id
        correct_id = arr_id.index(max(arr_id))
    else:
        # иначе следующий по порядку
        correct_id = global_id - count_same
    return correct_id


# метод детекции людей с помощью нейросети
# Input:
# frame - кадр с входного видео
# Output:
# координаты найденных боксов;
def net_find_box(frame, detections, size):
    outs_net = detector_openvino.forward(frame)  # результат работы нейронной сети
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
    outs_vector = detector_onnx.forward(frame)
    if np.all(np.abs(vector[global_id]) < EPSILON):
        vector[global_id] = outs_vector
    else:
        vector[global_id] = alpha * vector[global_id] + (1 - alpha) * outs_vector
    return


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


parser = argparse.ArgumentParser(description="Input video file")
parser.add_argument("video_path1", type=str, help="Path to the video file number 1")
parser.add_argument("video_path2", type=str, help="Path to the video file number 2")
parser.add_argument("openvino_path", type=str, help="Path to the openvino model")
parser.add_argument("onnx_path", type=str, help="Path to the onnx model")
args = parser.parse_args()

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

tracker1 = EuclideanDistTracker()
tracker2 = EuclideanDistTracker()

video_path1 = args.video_path1 + ".avi"
video_path2 = args.video_path2 + ".avi"
cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)

EPSILON = 1e-6
width, height = get_screen_resolution()
N = width * height

id2_save, id1_save = (-1, -1)
count_frame = 40  # количество кадров для идентификации человека
vector_size = 256  # размер выходного вектора из сети onnx

ret1, frame1_1 = cap1.read()
ret2, frame2_2 = cap2.read()
height_frame, width_frame, _ = frame2_2.shape

# минимальные и максимальные размеры боксов
size1 = (
    int(round(0.045 * width_frame * height_frame)),
    int(round(width_frame * height_frame / 3)),
)
size2 = (
    int(round(0.0076 * width_frame * height_frame)),
    int(round(width_frame * height_frame / 3)),
)

arr_id1, arr_id2 = (
    [0] * 10,
    [0] * 10,
)  # массив для накопления совпадений с конкретным объектом
vector = np.zeros((200, vector_size))
opt_param1, opt_param2 = (0.65, 0.48)  # границы сравнения векторов для каждого кадра
count_same = 0  # переменная для подсчета одинаковых объектов

report2 = True
output_frames = []

center1, center2 = ([0, 0], [0, 0])
time1, time2 = ([0, 0], [0, 0])

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # трекинг кадра с первой камеры
    detections1 = []
    detections2 = []
    # roi2 = frame2[0:height_frame, int(width_frame // 3.2): int(width_frame / 1.6)]

    net_find_box(frame1, detections1, size1)

    time1[1] = datetime.now().timestamp()
    # разница времени с последнего бокса в первом кадре с текущим моментом
    delta_time = int(time1[1]) - int(time1[0])

    boxes_ids1 = tracker1.update(detections1, center1, delta_time)
    report1 = True
    for box_id in boxes_ids1:
        x, y, w, h, id1 = box_id
        # если надпись окажется за пределами
        y1 = if_border(y, h)
        report1 = False
        frame_plt1 = frame1[y:h, x:w]
        # если новый объект
        if id1 != id1_save:
            # проверяем содержимое массива id для первой камеры и увеличиваем кол-во одинаковых объектов
            count_same = if_same_object(arr_id1, count_same, id1_save)
            # если нет объекта во второй камере
            if report2:
                # проверяем содержимое массива id для второй камеры и увеличиваем кол-во одинаковых объектов
                count_same = if_same_object(arr_id2, count_same, id2_save)
            # обновляем счетчик кадров для гистограмм и для детектора
            count_frame1 = count_frame

        global_id1 = id1 - count_same
        id1_save = id1

        # пока счетчик кадров больше нуля, считаем сходства новой гистограммы с уже имеющимися
        if count_frame1 > 0:
            # ищем вектор схожести
            net_search_vector(frame_plt1, global_id1)
            # search_compare(global_id1, opt_param1, arr_id1)
            net_count_compare(global_id1, opt_param1, arr_id1)
            count_frame1 -= 1
            correct_id1 = global_id1
        else:
            # иначе ищем id с макс совпадением
            correct_id1 = find_max_same_id(arr_id1, count_same, id1)
            id1 = correct_id1
            cv2.putText(
                frame1,
                "Object " + str(id1),
                (x, y1),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )
        center1 = center_point_save(x, w, y, h)
        cv2.rectangle(frame1, (x, y), (w, h), (255, 0, 0), 3)
        time1[0] = datetime.now().timestamp()
        tracker2.id_count = tracker1.id_count

    # трекинг кадра со второй камеры
    time2[1] = datetime.now().timestamp()
    # разница времени с последнего бокса во втором кадре с текущим моментом
    delta_time = int(time2[1]) - int(time2[0])
    report2 = True
    net_find_box(frame2, detections2, size2)
    boxes_ids2 = tracker2.update(detections2, center2, delta_time)
    for box_id in boxes_ids2:
        x, y, w, h, id2 = box_id
        # если надпись окажется за пределами
        y1 = if_border(y, h)
        frame_plt2 = frame2[y + 10 : h, x:w]
        report2 = False

        # если новый объект
        if id2 != id2_save:
            # проверяем содержимое массива id и увеличиваем кол-во одинаковых объектов
            count_same = if_same_object(arr_id2, count_same, id2_save)
            # если нет объекта в первой камере
            if report1:
                # проверяем содержимое массива id и увеличиваем кол-во одинаковых объектов
                count_same = if_same_object(arr_id1, count_same, id1_save)
            # обновляем счетчик кадров для гистограмм и для детектора
            count_frame2 = count_frame

        id2_save = id2
        global_id2 = id2 - count_same

        # пока счетчик кадров больше нуля, считаем сходства новой гистограммы с уже имеющимися
        if count_frame2 > 0:
            # ищем вектор схожести
            net_search_vector(frame_plt2, global_id2)
            # search_compare(global_id2, opt_param2, arr_id2)
            net_count_compare(global_id2, opt_param2, arr_id2)
            count_frame2 -= 1
            correct_id2 = global_id2
        else:
            # иначе ищем id с макс совпадением
            correct_id2 = find_max_same_id(arr_id2, count_same, id2)
            id2 = correct_id2
            cv2.putText(
                frame2,
                "Object " + str(id2),
                (x, y1),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )
        center2 = center_point_save(x, w, y, h)
        cv2.rectangle(frame2, (x, y), (w, h), (255, 0, 0), 3)
        tracker1.id_count = tracker2.id_count
        time2[0] = datetime.now().timestamp()
    resized_frame_1 = cv2.resize(frame1, (width // 2, height // 2))
    resized_frame_2 = cv2.resize(frame2, (width // 2, height // 2))
    combined_frame = cv2.hconcat([resized_frame_1, resized_frame_2])
    output_frames.append(combined_frame)
    cv2.imshow("combined", combined_frame)
    key = cv2.waitKey(30)
    if key == 27:
        break
cap1.release()
cap2.release()
cv2.destroyAllWindows()
