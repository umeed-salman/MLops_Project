import torch.optim as optim
from model import Net
from data import trainloader, testloader
import torch
import torch.nn as nn
import mlflow.pytorch


def train(net):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Log a summary of the model parameters
    param_summary = {name: param.numel() for name,
                     param in net.named_parameters()}
    mlflow.log_params(param_summary)

    # Train the network
    for epoch in range(10):  # Change the number of epochs if needed
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                # Log the loss metric
                mlflow.log_metric("loss",
                                  running_loss / 2000,
                                  step=i + epoch * len(trainloader))
                print('[%d, %5d] loss: %.3f' % (
                    epoch + 1,
                    i + 1,
                    running_loss / 2000
                ))
                running_loss = 0.0

    print('Finished training')


def test(net):
    # Test the network on the test dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and log the accuracy metric
    accuracy = 100 * correct / total
    mlflow.log_metric("accuracy", accuracy)
    print(
        'Accuracy of the network on the 10000 test images: %.2f %%' % accuracy)


if __name__ == '__main__':
    # Start an MLflow run
    with mlflow.start_run():
        net = Net()  # Create an instance of the neural network
        train(net)  # Train the network
        mlflow.pytorch.log_model(net, "model")
        test(net)   # Test the network
