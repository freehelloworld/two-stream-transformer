import numpy as np
import torch
from torch.autograd import Variable

from os.path import join
from glob import glob

import math
import torch
from two_stream_transformer import VisionTransformer
from pose_dataset import PoseDataset
from pose_dataset import PoseNomalise
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_results(result, fold_num):
    print('saving....')
    results = result.data.cpu().numpy()
    print(result.shape)
    lines = ['class1,class2,truth\n']
    for row in results:
        class1 = row[0]
        class2 = row[1]
        target = row[2]

        line = '{0}, {1}, {2}\n'.format(class1, class2, target)
        lines.append(line)
    name = 'late_result/fold{0}.csv'.format(fold_num)
    with open(name, 'w') as file:
        file.writelines(lines)


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, sample_batched in enumerate(dataloaders[phase]):
                context = sample_batched['context']
                pose = sample_batched['pose']
                target = sample_batched['label']
                context, pose, labels = context.to(device), pose.to(device).float(), target.to(device)

                # print('input:', inputs.shape)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(context, pose)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * pose.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # if batch_idx % 10 == 0:
                #     print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, running_loss, running_corrects))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test(model, device, test_loader, fold_num):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        results = []
        for batch_idx, sample_batched in enumerate(test_loader):
            context = sample_batched['context']
            pose = sample_batched['pose']
            target = sample_batched['label']
            context, pose, target = context.to(device), pose.to(device).float(), target.to(device)
            output = model(context, pose)

            postive_results = output[:, 1]
            negative_results = output[:, 0]

            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)

            result = torch.stack([negative_results, postive_results, target.float()], dim=-1)
            results.append(result)

            correct += pred.eq(target.view_as(pred)).sum().item()

        result = torch.cat(results, 0)
        save_results(result, fold_num)
    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PD pose graph')

    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 5)')

    parser.add_argument('--fold', type=int, default=1,
                        help='fold number to train (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    epochs = args.epochs
    fold = args.fold

    # source_file = '/home/rsun6573/work/pd_work/engineering_tools/pose_data_grouped.csv'
    source_file = '/home/rsun6573/work/pd_work/two_stream/updated_label.csv'

    train_data = PoseDataset(source_file, fold, True, transform=transforms.Compose([
        PoseNomalise(1)
    ]))

    test_data = PoseDataset(source_file, fold, False, transform=transforms.Compose([
        PoseNomalise(1)
    ]))

    train_loader = DataLoader(train_data, batch_size=256,
                              shuffle=True, num_workers=0)

    test_loader = DataLoader(test_data, batch_size=256,
                             shuffle=True, num_workers=0)

    dataset_sizes = {
        'train': len(train_data)
        ,
        'val': len(test_data)
    }

    dataloaders = {
        'train': train_loader
        ,
        'val': test_loader
    }

    torch.cuda.empty_cache()

    net = VisionTransformer()

    net.cuda()

    criterion = nn.CrossEntropyLoss()
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(net, criterion, optimizer_conv,
                             exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=epochs)

    test(model_conv, device, test_loader, fold)

    torch.save(model_conv.state_dict(), './f{0}_late.pickle'.format(fold))


# entry point
if __name__ == '__main__':
    main()
