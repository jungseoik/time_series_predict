import json
import os
import shutil
from datetime import datetime

def save_config_list(list15, list30, list60, list90):
    base_directory = "config_files"
    backup_directory = f"old_config_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if os.path.exists(base_directory):
        shutil.move(base_directory, backup_directory)

    os.makedirs(base_directory, exist_ok=True)

    config_lists = {
        "15일": list15,
        "30일": list30,
        "60일": list60,
        "90일": list90,
    }

    # Rest of your code for saving configurations
    for date, configs in config_lists.items():
        folder_path = os.path.join(base_directory, date)
        os.makedirs(folder_path, exist_ok=True)

        for i, config in enumerate(configs):
            config_dict = {
                "input_size": config[0],
                "hidden_size": config[1],
                "output_size": config[2],
                "layers": config[3],
                "period": config[4],
            }
            file_path = os.path.join(folder_path, f"config_{i}.json")
            with open(file_path, "w") as file:
                json.dump(config_dict, file, indent=4)

def load_config_list(str):
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
                    config_dict['input_size'],
                    config_dict['hidden_size'],
                    config_dict['output_size'],
                    config_dict['layers'],
                    config_dict['period'],
                    ]
                config_lists[date].append(config_list)

    # 결과 출력
    for date, configs in config_lists.items():
        print(f'날짜: {date}')
        for i, config in enumerate(configs):
            print(f'Config {i}: {config}')
            # list = config_lists['15일']
    
    print('selected_list', str)
    
    return config_lists[str]