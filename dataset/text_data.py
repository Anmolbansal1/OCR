from torch.utils.data import Dataset
import json
import os
import cv2

def create_config(data_path, abcd,mode):
    fname = 'annotation.txt'
    f = open(os.path.join(data_path, fname))
    lines = f.readlines()
    f.close()

    config = {}
    config['abc'] = abcd
    config['mode'] = mode
    config['train'] = []
    config['test'] = []
    samples = len(lines)
    train_samples = int(samples * 0.6)
    test_samples = samples - train_samples
    print('Total samples: ' + str(samples) + ' and train: '+ str(train_samples))
    for line in range(0, train_samples):
        path = lines[line].split(' ')[0]
        label = lines[line].split('_')[1]
        config['train'].append({
            "text": label,
            "name": path
    })

    # for line in range(train_samples+1, test_samples):
    #     path = lines[line].split(' ')[0]
    #     label = lines[line].split('_')[1]
    #     config['test'].append({
    #         "text": label,
    #         "name": path
    # })

    config['test'].append({
        "text": "ALTO",
        "name": "../../opencv-text-detection/img.png"
    })

    return config
    


class TextDataset(Dataset):
    def __init__(self, data_path, abc, mode="train", transform=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.config = create_config(data_path, abcd=abc, mode=mode)
        # self.config = json.load(open(os.path.join(data_path, "desc.json")))
        self.transform = transform
        # print(data_path + '\n' + mode + '\n' + str(transform) + '\n' + str(self.config))
    
        # print(self.config["test"])
    def abc_len(self):
        return len(self.config["abc"])

    def get_abc(self):
        return self.config["abc"]

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == "test":
            return int(len(self.config[self.mode]) * 0.01)
        return len(self.config[self.mode])

    def __getitem__(self, idx):
        name = self.config[self.mode][idx]["name"]
        text = self.config[self.mode][idx]["text"]

        # img = cv2.imread(os.path.join(self.data_path, "data", name))
        img = cv2.imread(os.path.join(self.data_path, name))
        cv2.imshow(img)
        cv2.waitKey(0)
        seq = self.text_to_seq(text)
        sample = {"img": img, "seq": seq, "seq_len": len(seq), "aug": self.mode == "train"}
        if self.transform:
            sample = self.transform(sample["img"])
        return sample

    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.config["abc"].find(c) + 1)
        return seq
