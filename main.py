import cv2
import numpy as np
from openvino.inference_engine import IECore
import pyautogui


# метод определения размера экрана
# Output:
# ширина и высота экрана
def get_screen_resolution():
    screen_width, screen_height = pyautogui.size()
    return screen_width, screen_height


# метод детекции людей с помощью нейросети
# Input:
# frame - кадр с входного видео
# Output:
# 1) при наличии человека - кадр с боксом;
# 2) при отсутствии человека - исходный кадр.
def net_find_box(frame):
    input_size = (300, 300)  # размер для изменения входного изображения
    resized_frame = cv2.resize(frame,
                         input_size,
                         interpolation=cv2.INTER_AREA)  # уменьшение изображения до размера входа сети
    input_frame = resized_frame.transpose(2, 0, 1)  # изменение RGB -> BGR
    input_name = next(iter(exec_net.input_info))  # функция синхронного исполнения нейронной сети (блокирует пользовательское приложение на время выполнения запроса на вывод)
    output = exec_net.infer({input_name: input_frame})  # инференс нейронной сети
    outs_net = next(iter(output.values()))  # выходные данные нейронной сети

    outs_net = outs_net[0][0]
    for out in outs_net:
        coord_array = []
        if out[2] == 0.0:
            break
        if out[2] > size_out:
            coord_array.append([out[3],
                           out[4],
                           out[5],
                           out[6]])  # добавление в массив координат

            coord_full_size = coord_array[0]
            h = frame.shape[0]
            w = frame.shape[1]
            coord_full_size = coord_full_size * np.array([w, h, w, h])  # преобразование относительных координат в абсолютные на исходном изображении
            coord_full_size = coord_full_size.astype(np.int32)  # возвращает копию массива, преобразованного к указанному типу (вещественные числа с одинарной точностью)
            cv2.rectangle(frame, (coord_full_size[0], coord_full_size[1]), (coord_full_size[2], coord_full_size[3]), color=(255, 0, 0), thickness=2)  # отрисовка бокса
        return


# Путь к файлам модели и весам
model_path_bin = 'GeneralNMHuman_v1.0_IR10_FP16.bin'
model_path_xml = 'GeneralNMHuman_v1.0_IR10_FP16.xml'

# создание объекта IECore
ie = IECore()
# загрузка нейронной сети из файлов и выбор устройства для инференса
net = ie.read_network(model=model_path_xml, weights=model_path_bin)
exec_net = ie.load_network(network=net, device_name='CPU')

size_out = 0.4  # минимальная уверенность сети
video_path1 = '3.Camera.avi'
video_path2 = '4.Camera.avi'
cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)
width, height = get_screen_resolution()
output_frames = []

while cap1.isOpened():
    ret, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret:
        break
    # Обработка кадров с разных камер сетью
    net_find_box(frame1)
    net_find_box(frame2)

    # Отображение кадров с обеих камер
    # уменьшение размера каждого кадра относительно экрана
    resized_frame_1 = cv2.resize(frame1, (width // 2, height // 2))
    resized_frame_2 = cv2.resize(frame2, (width // 2, height // 2))
    # слияние двух кадров в один
    combined_frame = cv2.hconcat([resized_frame_1, resized_frame_2])
    output_frames.append(combined_frame)
    # отображение результата
    cv2.imshow("combined", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cv2.destroyAllWindows()