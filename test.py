import torch
import torchvision
from torchvision import models

from model import trans_resnet_50
from datasets import load_test_data, submission_trans

from tqdm.auto import tqdm

def main():
    dev = torch.device('cpu')
    if torch.cuda.is_available:
        dev = torch.device('cuda:0')
    
    test_loader, submission = load_test_data()
    submission = test(test_loader, submission, dev)
    submission.to_csv("./submission.csv", index=False)
    #print(submission)
    print('[Datas has been classified into submission.csv]')
    print('[Test Complete]')

def test(test_loader, submission, dev):
    model = trans_resnet_50()
    weights = torch.load('./trained_model_weight.pth', map_location='cpu')
    model.load_state_dict(weights)
    model.to(dev)

    model.eval()
    for img, file in tqdm(test_loader):
        img = img.to(dev)
        predict = model(img)
        _, predict_class = torch.max(predict, dim=1)
        #print(type(predict_class))
        predict_class = predict_class.cpu().numpy()
        
        for i in range(len(predict_class)):
            for file_index in range(len(submission)):
                if submission.iloc[file_index,0] == file[i]:
                    temp = int(predict_class[i])
                    submission.iloc[file_index,1] = temp

    submission = submission_trans(submission)
    
    return submission


if __name__ == '__main__':
    main()
