import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


def message_gpu_usage(train_on_gpu):
    if train_on_gpu:
        return 'CUDA is available!  Training on GPU ...'
    else:
        return 'CUDA is not available.  Training on CPU ...'


# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layers
        # 32x32x3->16x16x3
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 16x16x3->8x8x3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 8x8x3->4x4x3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # linear:
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

        # dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = self.pool(f.relu(self.conv3(x)))

        x = x.view(-1, 64 * 4 * 4)

        x = self.dropout(x)
        x = f.relu(self.fc1(x))

        x = self.dropout(x)
        x = f.relu(self.fc2(x))

        return x


class CIFAR10CNNClassifier:
    def __init__(self, num_workers=0, batch_size=20, valid_size=0.2):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.valid_size = 0.2

        # convert data to a normalized torch.FloatTensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # choose the training and test datasets
        train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

        # obtain training indices that will be used for validation
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # prepare data loaders (combine dataset and sampler)
        self.train_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=num_workers)
        self.valid_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(test_data,
                                                       batch_size=batch_size,
                                                       num_workers=num_workers)

        # specify the image classes
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def train(self, model, criterion, train_on_gpu):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            model.cuda()

        # specify optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # number of epochs to train the model
        n_epochs = 8  # you may increase this number to train a final model

        valid_loss_min = np.Inf  # track change in validation loss

        for epoch in range(1, n_epochs + 1):

            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0

            ###################
            # train the model #
            ###################
            model.train()
            for data, target in self.train_loader:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)

            ######################
            # validate the model #
            ######################
            model.eval()
            for data, target in self.valid_loader:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                valid_loss += loss.item() * data.size(0)

            # calculate average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            valid_loss = valid_loss / len(self.valid_loader.dataset)

            # print training/validation statistics
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.state_dict(), 'models/model_cifar.pt')
                valid_loss_min = valid_loss

    def test(self, model, criterion, train_on_gpu):

        # track test loss
        test_loss = 0.0
        class_correct = list(0. for _ in range(10))
        class_total = list(0. for _ in range(10))

        model.eval()
        # iterate over test data
        for data, target in self.test_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update test loss
            test_loss += loss.item() * data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)

            # compare predictions to true label
            def correct(t):
                t = t.eq(target.data.view_as(t))
                if train_on_gpu:
                    t = t.cpu().numpy()
                return np.squeeze(t)

            correct = correct(pred)
            # calculate test accuracy for each object class
            for i in range(self.batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        # average test loss
        test_loss = test_loss / len(self.test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        for i in range(10):
            if class_total[i] > 0:
                # noinspection PyStringFormat
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    self.classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (self.classes[i]))

        # noinspection PyStringFormat
        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))


def main():

    train_on_gpu = torch.cuda.is_available()

    print(message_gpu_usage(train_on_gpu))

    # create a complete CNN
    model = Net()
    print(model)

    cifar10cnn = CIFAR10CNNClassifier()

    # specify loss function
    criterion = nn.CrossEntropyLoss()

#    cifar10cnn.train(model, criterion, train_on_gpu)

    best_model = Net()
    if train_on_gpu:
        best_model.cuda()
    best_model.load_state_dict(torch.load('models/model_cifar.pt'))

    cifar10cnn.test(best_model, criterion, train_on_gpu)


# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


def visualize_batch_of_training_data(train_loader, classes):
    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()  # convert images to numpy for display

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])


def view_image_in_detail(img):
    rgb_img = np.squeeze(img)
    channels = ['red channel', 'green channel', 'blue channel']

    fig = plt.figure(figsize=(36, 36))
    for idx in np.arange(rgb_img.shape[0]):
        ax = fig.add_subplot(1, 3, idx + 1)
        img = rgb_img[idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(channels[idx])
        width, height = img.shape
        thresh = img.max() / 2.5
        for x in range(width):
            for y in range(height):
                val = round(img[x][y], 2) if img[x][y] != 0 else 0
                ax.annotate(str(val), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center', size=8,
                            color='white' if img[x][y] < thresh else 'black')


def visualize_sample_test_result(test_loader, train_on_gpu, model, classes):
    # obtain one batch of test images
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images.numpy()

    # move model inputs to cuda, if GPU available
    if train_on_gpu:
        images = images.cuda()

    # get sample outputs
    output = model(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    images = images.cpu()
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                     color=("green" if preds[idx] == labels[idx].item() else "red"))


if __name__ == '__main__':
    main()
