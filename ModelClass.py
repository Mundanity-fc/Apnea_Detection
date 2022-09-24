from typing import Tuple, Any
import yaml
import os
import numpy
import scipy.io as scipy_io
from keras.utils.vis_utils import plot_model
from keras.models import *
from keras.layers import *
from numpy import ndarray
import pickle


class ModelClass:
    """
    模型调用类
    可使用的成员函数有：
    
    """""

    workdir = os.path.dirname(os.path.abspath(__file__)) + '/'

    def __init__(self):
        # 获取yaml配置文件
        config_file = open(self.workdir+'config/config.yaml', 'r', encoding='utf-8')
        config_content = config_file.read()
        self.model = ''
        self.config = yaml.load(config_content, Loader=yaml.FullLoader)
        # 定义标签
        self.label = ['normal', 'apnea']

    def normalize(self, data):
        mean = numpy.mean(data)
        max = numpy.max(data)
        min = numpy.min(data)
        normal = [(i - mean) / (max - min) for i in data]
        return numpy.array(normal, dtype=numpy.float32)

    def init_train_data(self) -> Tuple[ndarray, ndarray]:
        """
        初始化训练集
        :return: x_train—数据集 y_train-标签集
        """""
        train_dir = self.config['dataset']['train_dir']
        normal_list = os.listdir(self.workdir + train_dir + "normal/")
        apnea_list = os.listdir(self.workdir + train_dir + "apnea/")
        x_train = []
        y_train = []
        for x in normal_list:
            if x.endswith('.mat'):
                data = scipy_io.loadmat(self.workdir + train_dir + "normal/" + x)["data"]
                x_train.append(self.normalize(data))
                y_train.append([1, 0])
        for x in apnea_list:
            if x.endswith('.mat'):
                data = scipy_io.loadmat(self.workdir + train_dir + "apnea/" + x)["data"]
                x_train.append(self.normalize(data))
                y_train.append([0, 1])
        y_train = list(y_train)
        x_train = numpy.array(x_train, dtype=numpy.float32)
        y_train = numpy.asarray(y_train)
        return x_train, y_train

    def init_test_data(self) -> Tuple[ndarray, ndarray]:
        """
        初始化测试集
        :return: x_test—数据集 y_test-标签集
        """""
        test_dir = self.config['dataset']['test_dir']
        normal_list = os.listdir(self.workdir + test_dir + "normal/")
        apnea_list = os.listdir(self.workdir + test_dir + "apnea/")
        x_test = []
        y_test = []
        for x in normal_list:
            if x.endswith('.mat'):
                data = scipy_io.loadmat(self.workdir + test_dir + "normal/" + x)["data"]
                x_test.append(self.normalize(data))
                y_test.append([1, 0])
        for x in apnea_list:
            if x.endswith('.mat'):
                data = scipy_io.loadmat(self.workdir + test_dir + "apnea/" + x)["data"]
                x_test.append(self.normalize(data))
                y_test.append([0, 1])
        y_test = list(y_test)
        x_test = numpy.array(x_test, dtype=numpy.float32)
        y_test = numpy.asarray(y_test)
        return x_test, y_test

    def init_validation_data(self) -> Tuple[ndarray, ndarray]:
        """
        初始化验证集
        :return: x_vali—数据集 y_vali-标签集
        """""
        validation_dir = self.config['dataset']['validation_dir']
        normal_list = os.listdir(self.workdir + validation_dir + "normal/")
        apnea_list = os.listdir(self.workdir + validation_dir + "apnea/")
        x_vali = []
        y_vali = []
        for x in normal_list:
            if x.endswith('.mat'):
                data = scipy_io.loadmat(self.workdir + validation_dir + "normal/" + x)["data"]
                x_vali.append(self.normalize(data))
                y_vali.append([1, 0])
        for x in apnea_list:
            if x.endswith('.mat'):
                data = scipy_io.loadmat(self.workdir + validation_dir + "apnea/" + x)["data"]
                x_vali.append(self.normalize(data))
                y_vali.append([0, 1])
        y_vali = list(y_vali)
        x_vali = numpy.array(x_vali, dtype=numpy.float32)
        y_vali = numpy.asarray(y_vali)
        return x_vali, y_vali

    def start_train(self):
        """
        进行模型的训练
        数据集、模型保存位置等变量定义于configs.yaml中
        :return: 无返回值
        """""
        # 获取训练集与验证集
        x_train, y_train = self.init_train_data()
        x_vali, y_vali = self.init_validation_data()
        # 定义输入层（通道数及名称）
        inputs = Input(shape=(60000, 1), name='inputs')
        # 定义第一卷积层，输入为输入层数据
        Layer_1_Conv1 = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', name="Layer_1_Convolution1")(
            inputs)
        Layer_1_Conv2 = Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', name="Layer_1_Convolution2")(
            Layer_1_Conv1)
        Layer_1_BatchNorm = BatchNormalization()(Layer_1_Conv2)
        Layer_1_Pool = MaxPooling1D(pool_size=2, strides=2)(Layer_1_BatchNorm)
        Layer_1_Drop = Dropout(rate=0.3)(Layer_1_Pool)

        Layer_2_Conv1 = Conv1D(filters=128, kernel_size=7, strides=1, activation='relu', name="Layer_2_Convolution1")(
            Layer_1_Drop)
        Layer_2_Conv2 = Conv1D(filters=256, kernel_size=7, strides=1, activation='relu', name="Layer_2_Convolution2")(
            Layer_2_Conv1)
        Layer_2_BatchNorm = BatchNormalization()(Layer_2_Conv2)
        Layer_2_Pool = MaxPooling1D(pool_size=2, strides=2)(Layer_2_BatchNorm)
        Layer_2_Drop = Dropout(rate=0.3)(Layer_2_Pool)

        Layer_3_Conv1 = Conv1D(filters=256, kernel_size=7, strides=1, activation='relu', name="Layer_3_Convolution1")(
            Layer_2_Drop)
        Layer_3_Conv2 = Conv1D(filters=64, kernel_size=7, strides=1, activation='relu', name="Layer_3_Convolution2")(
            Layer_3_Conv1)
        Layer_3_BatchNorm = BatchNormalization()(Layer_3_Conv2)
        Layer_3_Pool = MaxPooling1D(pool_size=2, strides=2)(Layer_3_BatchNorm)
        Layer_3_Drop = Dropout(rate=0.3)(Layer_3_Pool)

        Layer_4_Conv1 = Conv1D(filters=256, kernel_size=7, strides=1, activation='relu', name="Layer_4_Convolution1")(
            Layer_3_Drop)
        Layer_4_Conv2 = Conv1D(filters=64, kernel_size=7, strides=1, activation='relu', name="Layer_4_Convolution2")(
            Layer_4_Conv1)
        Layer_4_BatchNorm = BatchNormalization()(Layer_4_Conv2)
        Layer_4_Pool = MaxPooling1D(pool_size=2, strides=2)(Layer_4_BatchNorm)
        Layer_4_Drop = Dropout(rate=0.3)(Layer_4_Pool)

        Layer_5_Conv1 = Conv1D(filters=8, kernel_size=2, strides=1, activation='relu', name="Layer_5_Convolution1")(
            Layer_4_Drop)
        Layer_5_Conv2 = Conv1D(filters=8, kernel_size=2, strides=1, activation='relu', name="Layer_5_Convolution2")(
            Layer_5_Conv1)
        Layer_5_BatchNorm = BatchNormalization()(Layer_5_Conv2)
        Layer_5_Pool = MaxPooling1D(pool_size=2, strides=2)(Layer_5_BatchNorm)
        Layer_5_Drop = Dropout(rate=0.3)(Layer_5_Pool)

        Layer_Flatten = Flatten()(Layer_5_Drop)
        Layer_Dense1 = Dense(64, activation='relu')(Layer_Flatten)
        Layer_Drop1 = Dropout(rate=0.3)(Layer_Dense1)
        Layer_Dense2 = Dense(32, activation='relu')(Layer_Drop1)
        Layer_Drop2 = Dropout(rate=0.3)(Layer_Dense2)

        outs = Dense(2, activation='softmax')(Layer_Drop2)

        # 定义模型的输入与输出
        model = Model(inputs=inputs, outputs=outs)
        model.compile(optimizer=self.config['model']['optimizer'], loss=self.config['model']['loss'],
                      metrics=['accuracy'])
        # 控制台输出模型的摘要
        print(model.summary())
        # 导出模型的结构
        plot_model(model, to_file=self.workdir + 'asset/model_structure.jpg', show_shapes=True)
        # 开始训练模型
        history = model.fit(x_train,
                            y_train,
                            batch_size=self.config['model']['batch'],
                            epochs=self.config['model']['epochs'],
                            verbose=2,
                            validation_data=(x_vali, y_vali))
        # 定义模型文件名与记录文件名
        filename_str = self.workdir + '{}new_trained_{}_{}_bs_{}_epochs_{}{}'
        current_model_file = filename_str.format(self.config['model']['model_dir'],
                                                 self.config['model']['optimizer'],
                                                 self.config['model']['loss'],
                                                 self.config['model']['batch'],
                                                 self.config['model']['epochs'],
                                                 self.config['model']['model_format'])
        history_file = filename_str.format(self.config['model']['history_dir'],
                                           self.config['model']['optimizer'],
                                           self.config['model']['loss'],
                                           self.config['model']['batch'],
                                           self.config['model']['epochs'],
                                           self.config['model']['history_format'])
        model.save(current_model_file)
        print('已将模型文件保存到：%s ' % current_model_file)
        print(history.history['accuracy'])
        print(history.history.keys())
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)
        print('已将模型记录保存到：%s ' % history_file)

    def predict_testset(self) -> str:
        """
        :return: 预测的结果
        """""
        filename_str = self.workdir + '{}new_trained_{}_{}_bs_{}_epochs_{}{}'
        model = load_model(filename_str.format(self.config['model']['model_dir'],
                                               self.config['model']['optimizer'],
                                               self.config['model']['loss'],
                                               self.config['model']['batch'],
                                               self.config['model']['epochs'],
                                               self.config['model']['model_format']))
        x_test, y_test = self.init_test_data()
        predict = model.predict(x_test)
        predict = predict.tolist()
        result = []
        for x in predict:
            if x[0] >= x[1]:
                result.append([1, 0])
            else:
                result.append([0, 1])
        result = numpy.asarray(result)
        sum = len(result)
        correct = 0
        for x in range(sum):
            if y_test[x] == result[x]:
                correct += 1
        acc = correct / sum
        print(acc)
        return acc

    def load_model(self):
        filename_str = self.workdir + '{}new_trained_{}_{}_bs_{}_epochs_{}{}'
        self.model = load_model(filename_str.format(self.config['model']['model_dir'],
                                                    self.config['model']['optimizer'],
                                                    self.config['model']['loss'],
                                                    self.config['model']['batch'],
                                                    self.config['model']['epochs'],
                                                    self.config['model']['model_format']))

    def target_detect(self, path):
        x = []
        data = scipy_io.loadmat(path)["data"]
        x.append(self.normalize(data))
        x = numpy.array(x, dtype=numpy.float32)
        predict = self.model.predict(x)
        predict = predict.tolist()
        if predict[0][0] >= predict[0][1]:
            result = '正常呼吸'
        else:
            result = '呼吸暂停'
        return result