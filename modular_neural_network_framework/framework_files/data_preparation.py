import numpy as np
import pathlib
import csv
import struct

class Iris:

    def load_Iris_data(file_path):

        label_map = {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2,
            }

        features = []
        labels = []
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)

            for row in reader:
                if not row:
                    continue

                features.append([float(x) for x in row[:-1]])
                labels.append(label_map[row[-1]])

        return np.array(features, dtype=np.float32), np.eye(3, dtype=np.float32)[labels]


    def Iris_data():
        file_path = pathlib.Path(r"/home/bishmay/AI/data/iris/iris.data")

        # READ DATASET
        input_data, target = Iris.load_Iris_data(file_path)

        # SPLIT DATASET
        rng = np.random.default_rng(23)

        train_indices = []
        validation_indices = []
        test_indices = []

        target_labels = np.argmax(target, axis=1)

        for class_label in range(3):
            
            class_indices = np.where(target_labels == class_label)[0]
            rng.shuffle(class_indices)

            train_indices.extend(class_indices[:30])
            validation_indices.extend(class_indices[30:40])
            test_indices.extend(class_indices[40:])

        rng.shuffle(train_indices)
        rng.shuffle(validation_indices)
        rng.shuffle(test_indices)

        x_train_input = input_data[train_indices]
        x_validation_input = input_data[validation_indices]
        x_test_input = input_data[test_indices]

        std_x = np.std(x_train_input, axis=0)
        mean = np.mean(x_train_input, axis=0)
        std = np.where(std_x == 0, 1, std_x)

        train_input = (x_train_input - mean) / std
        train_target = target[train_indices]

        validation_input = (x_validation_input - mean) / std
        validation_target = target[validation_indices]

        test_input = (x_test_input - mean) / std
        test_target = target[test_indices]

        train_data = [(train_input, train_target)]
        validation_data = [(validation_input, validation_target)]
        test_data = [(test_input, test_target)]

        return train_data, validation_data, test_data

class MNIST:

    def load_mnist_images(file_path):
        with open(file_path, "rb") as file:
            magic_number, number_images, rows, columns = struct.unpack(">IIII", file.read(16))

            if magic_number != 2051:
                raise ValueError(f"Invalid magic number {magic_number} in MNIST image file: {file_path}")

            images = np.frombuffer(file.read(), dtype=np.uint8).reshape(number_images, rows * columns)

        return images

    def load_mnist_labels(file_path):
        with open(file_path, "rb") as file:
            magic_number, number_labels = struct.unpack(">II", file.read(8))

            if magic_number != 2049:
                raise ValueError(f"Invalid magic number {magic_number} in MNIST label file: {file_path}")

            labels = np.frombuffer(file.read(), dtype=np.uint8)

            if len(labels) != number_labels:
                raise ValueError(f"Expected {number_labels} labels, but got {len(labels)} in MNIST label file: {file_path}")

        return labels

    def MNIST_data():
        file_path = pathlib.Path(r"/home/bishmay/AI/data/MNIST")

        MNIST_data_path = {
            "train_input_path": file_path / "train-images.idx3-ubyte",
            "train_target_path": file_path / "train-labels.idx1-ubyte",
            "test_input_path": file_path / "t10k-images.idx3-ubyte",
            "test_target_path": file_path / "t10k-labels.idx1-ubyte"
        }

        # READ DATASET
        z_train_input = MNIST.load_mnist_images(MNIST_data_path["train_input_path"])
        z_train_target = MNIST.load_mnist_labels(MNIST_data_path["train_target_path"])

        z_test_input = MNIST.load_mnist_images(MNIST_data_path["test_input_path"])
        z_test_target = MNIST.load_mnist_labels(MNIST_data_path["test_target_path"])

        # SHUFFLING & SPLITTING
        rng = np.random.default_rng(23)

        train_indices = rng.permutation(len(z_train_input))
        test_indices = rng.permutation(len(z_test_target))

        train_input_indices = z_train_input[train_indices]
        train_target_indices = z_train_target[train_indices]

        x_train_input = train_input_indices[:50000]
        x_train_target = train_target_indices[:50000]

        x_validation_input = train_input_indices[50000:]
        x_validation_target = train_target_indices[50000:]

        x_test_input = z_test_input[test_indices]
        x_test_target = z_test_target[test_indices]

        train_input = x_train_input.astype(np.float32) / 255.0
        train_target = np.eye(10, dtype=np.float32)[x_train_target]

        validation_input = x_validation_input.astype(np.float32) / 255.0
        validation_target = np.eye(10, dtype=np.float32)[x_validation_target]

        test_input = x_test_input.astype(np.float32) / 255.0
        test_target = np.eye(10, dtype=np.float32)[x_test_target]

        train_data = [(train_input, train_target)]
        validation_data = [(validation_input, validation_target)]
        test_data = [(test_input, test_target)]

        return train_data, validation_data, test_data