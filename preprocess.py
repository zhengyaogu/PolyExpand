import json
from data import Lang

def txt_to_json(txt_path, json_path):
    l = []
    with open(txt_path, 'r') as f:
        line = f.readline()
        while line:
            split_line = line[:-1].split("=")
            l.append(split_line)
            line = f.readline()
    
    with open(json_path, 'w') as f:
        json.dump(l, f)

def save_language(data_path, save_path):
    lang = Lang.construct_from_json(data_path)
    with open(save_path, 'w') as f:
        json.dump(lang.export(), f)
    
if __name__ == "__main__":
    save_language('data/data.json', 'data/lang.json')