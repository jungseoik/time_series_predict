import json
import os


def config_list(str):
        # 설정값을 저장할 기본 디렉토리
    # base_directory = 'config_files'

    current_directory = os.path.dirname(__file__)
    base_directory = os.path.join(current_directory, 'config_files')

    # 날짜별 폴더 이름
    dates = ['15일', '30일', '60일', '90일']

    # 빈 딕셔너리를 초기화한 config_lists
    config_lists = {
        '15일': [],
        '30일': [],
        '60일': [],
        '90일': []
    }

    # 각 날짜별로 설정값 파일을 읽어와서 config_lists에 추가
    for date in dates:
        folder_path = os.path.join(base_directory, date)
        for i in range(len(os.listdir(folder_path))):
            file_path = os.path.join(folder_path, f'config_{i}.json')
            with open(file_path, 'r') as file:
                config_dict = json.load(file)
                config_list = [
                    config_dict['patch_length'],
                    config_dict['n_patches'],
                    config_dict['prediction_length'],
                    config_dict['tst_size'],
                    config_dict['model_dim'],
                    config_dict['heads'],
                    config_dict['layer'],
                    config_dict['epoch']
                ]
                config_lists[date].append(config_list)

    # 결과 출력
    for date, configs in config_lists.items():
        print(f'날짜: {date}')
        for i, config in enumerate(configs):
            print(f'Config {i}: {config}')

    list = config_lists['15일']

    return config_lists[str]

