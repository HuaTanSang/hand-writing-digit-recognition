import numpy as np
import idx2numpy

# Dataset class for MNIST
class MNISTDataset:
    def __init__(self, image_path, label_path) -> None:
        super().__init__()

        # Chuyển đổi ảnh và nhãn từ định dạng IDX sang NumPy
        images = idx2numpy.convert_from_file(image_path).tolist()
        labels = idx2numpy.convert_from_file(label_path).tolist()

        self.__data = {}

        for ith, (image, label) in enumerate(zip(images, labels)):
            self.__data[ith] = {
                "image": np.array(image),  # 28 x 28
                "label": label   # Nhãn lớp
            }

    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self, index: int) -> dict:
        return self.__data[index]
