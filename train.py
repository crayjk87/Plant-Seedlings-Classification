import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from model import trans_resnet_50
from datasets import load_train_data

from utils import save_plot
from utils.save_model import save_model

from tqdm.auto import tqdm

def main():
    epoch_num =100
    dev = torch.device('cpu')
    if torch.cuda.is_available:
        dev = torch.device('cuda:0')

    train_loader, val_loader = load_train_data()
    model = trans_resnet_50()
    model_trained = train_res(model, dev, epoch_num, train_loader, val_loader)
    save_model(model_trained)

    print('[Training Completed]')


def train_res(model, dev, epoch_num, train_loader, val_loader):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    model.to(dev)
    train_total_loss, val_total_loss = [], []
    train_total_acc, val_total_acc = [], []

    for epoch in range(epoch_num):

        print(f'>> Epoch {epoch+1} of {epoch_num}')
        training_loss = 0.0
        training_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        model.train()
        print('[Training]')
        for img, label in tqdm(train_loader):
            img = img.to(dev)
            label = label.to(dev)
            
            output = model(img)
            loss = criterion(output, label)
            training_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predict_class = output.argmax(dim=1, keepdim=True)

            #print(len(predict_class))
            training_acc += predict_class.eq(label.view_as(predict_class)).sum().item()/len(predict_class) # one batch is 32

        training_loss /= len(train_loader)
        training_acc /= len(train_loader)

        train_total_acc.append(training_acc)
        train_total_loss.append(training_loss)

        model.eval()
        print('[Validating]')
        for img, label in tqdm(val_loader):
            img = img.to(dev)
            label = label.to(dev)

            output = model(img)
            loss = criterion(output, label)
            val_loss += loss.item()

            predict_class = output.argmax(dim=1, keepdim=True)
            
            val_acc += predict_class.eq(label.view_as(predict_class)).sum().item()/len(predict_class)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        val_total_acc.append(val_acc)
        val_total_loss.append(val_loss)

        #model.load_state_dict(model.state_dict())


        print(f'Training loss: {training_loss:.3f}, training acc:{training_acc:.3f}')
        print(f'Validation loss: {val_loss:.3f}, validation acc:{val_acc:.3f}')
        print('\n')
    
    save_plot.save_acc(train_total_acc, val_total_acc)
    save_plot.save_loss(train_total_loss, val_total_loss)

    return model



if __name__ == '__main__':
    main()