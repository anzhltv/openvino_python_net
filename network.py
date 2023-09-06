from openvino.inference_engine import IECore
import cv2


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


    def data_preparation(self, frame):
        resized_frame = cv2.resize(
            frame, self.input_size, interpolation=cv2.INTER_AREA
        )  ## изменение изображения до размера входа сети
        input_frame = resized_frame.transpose(2, 0, 1)  # изменение RGB -> BGR
        return input_frame


    def forward(self, input_frame):
        input_name = next(
            iter(self.exec_net.input_info)
        )  # функция синхронного исполнения нейронной сети (блокирует пользовательское приложение на время выполнения запроса на вывод)
        output = self.exec_net.infer(
            {input_name: input_frame}
        )  # инференс нейронной сети
        outs_net = next(iter(output.values()))  # выходные данные нейронной сети
        return outs_net