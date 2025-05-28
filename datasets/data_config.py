class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"

    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'data_levir':
            self.root_dir = 'E://遥感图像变化检测//data_levir//'
        elif data_name == 'data_CDD':
            self.root_dir = 'E://遥感图像变化检测//data_CDD//'
        elif data_name == 'quick_start':
            self.root_dir = './samples_v/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)
