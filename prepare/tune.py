import json
import os

def read(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)
    print('done', file_path, len(data))

def get_file_path(mode, dataset):
    file_dir = 'data/depparsed'
    dataset_map = {
        'train': {
            'rest': 'Restaurants_Train_v2',
            'laptop': 'Laptops_Train_v2'
        },
        'test': {
            'rest': 'Restaurants_Test_Gold',
            'laptop': 'Laptops_Test_Gold'
        }
    }
    file_name=dataset_map[mode][dataset]+'_Biaffine_depparsed.json'
    return os.path.join(file_dir, dataset, file_name)

def main():
    file_paths=[
        get_file_path('train', 'rest'),
        get_file_path('train', 'laptop'),
        get_file_path('test', 'rest'),
        get_file_path('test', 'laptop'),
    ]

    for file_path in file_paths:
        docs=read(file_path)
        for doc in docs:
            doc["predicted_dependencies"] = [x[0] for x in doc["dependencies"]]
            doc["predicted_heads"] = [x[1] for x in doc["dependencies"]]
        write(file_path, docs)

if __name__ == "__main__":
    main()