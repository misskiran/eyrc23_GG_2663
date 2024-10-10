import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim


def data_preprocessing(task_1a_dataframe):
    label_encoder = LabelEncoder()

    for col in task_1a_dataframe.columns:
        if task_1a_dataframe[col].dtype == 'object':
            task_1a_dataframe[col] = label_encoder.fit_transform(task_1a_dataframe[col])

    return task_1a_dataframe


def identify_features_and_targets(encoded_dataframe):
    '''
    Purpose:
    ---
    The purpose of this function is to define the features and
    the required target labels. The function returns a python list
    in which the first item is the selected features and second
    item is the target label

    Input Arguments:
    ---
    encoded_dataframe : [ Dataframe ]
                        Pandas dataframe that has all the features mapped to
                        numbers starting from zero

    Returns:
    ---
    features_and_targets : [ list ]
                            python list in which the first item is the
                            selected features and second item is the target label
    '''

    features = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched',
                'ExperienceInCurrentDomain']
    targets = ['LeaveOrNot']

    features_and_targets = [encoded_dataframe[features], encoded_dataframe[targets]]

    return features_and_targets


def load_as_tensors(features_and_targets):
    '''
    Purpose:
    ---
    This function aims at loading your data (both training and validation)
    as PyTorch tensors. Here you will have to split the dataset for training
    and validation, and then load them as as tensors.
    Training of the model requires iterating over the training tensors.
    Hence the training tensors need to be converted to iterable dataset
    object.

    Input Arguments:
    ---
    features_and_targets : [ list ]
                            python list in which the first item is the
                            selected features and second item is the target label

    Returns:
    ---
    tensors_and_iterable_training_data : [ list ]
                                        Items:
                                        [0]: X_train_tensor: Training features loaded into Pytorch array
                                        [1]: X_test_tensor: Feature tensors in validation data
                                        [2]: y_train_tensor: Training labels as Pytorch tensor
                                        [3]: y_test_tensor: Target labels as tensor in validation data
                                        [4]: Iterable dataset object and iterating over it in
                                             batches, which are then fed into the model for processing
    '''

    X, y = features_and_targets

    X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y.values, dtype=torch.float32)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X_train_tensor[:split_idx], X_train_tensor[split_idx:]
    y_train, y_test = y_train_tensor[:split_idx], y_train_tensor[split_idx:]

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    validation_data = TensorDataset(X_test, y_test)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False)

    tensors_and_iterable_training_data = [X_train, X_test, y_train, y_test, train_loader, validation_loader]

    return tensors_and_iterable_training_data


class Salary_Predictor(nn.Module):
    def __init__(self):
        super(Salary_Predictor, self).__init__()
        '''
        Define the type and number of layers
        '''

        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.fc5 = nn.Linear(1, 1)  # Additional layer with 1 output unit

    def forward(self, x):
        '''
        Define the activation functions
        '''
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))  # Updated activation function
        x = torch.sigmoid(self.fc5(x))  # Sigmoid activation in the final layer
        return torch.round(x)


# ----------------------#After Building Model-----------------------------#
def model_loss_function():
    '''
    Purpose:
    ---
    To define the loss function for the model. Loss function measures
    how well the predictions of a model match the actual target values
    in training data.

    Input Arguments:
    ---
    None

    Returns:
    ---
    loss_function: This can be a pre-defined loss function in PyTorch
                    or can be user-defined

    Example call:
    ---
    loss_function = model_loss_function()
    '''

    loss_function = nn.BCELoss()

    return loss_function


def model_optimizer(model):
    '''
    Purpose:
    ---
    To define the optimizer for the model. Optimizer is responsible
    for updating the parameters (weights and biases) in a way that
    minimizes the loss function.

    Input Arguments:
    ---
    model: An object of the 'Salary_Predictor' class

    Returns:
    ---
    optimizer: Pre-defined optimizer from PyTorch

    Example call:
    ---
    optimizer = model_optimizer(model)
    '''

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return optimizer


def model_number_of_epochs():
    '''
    Purpose:
    ---
    To define the number of epochs for training the model

    Input Arguments:
    ---
    None

    Returns:
    ---
    number_of_epochs: [integer value]

    Example call:
    ---
    number_of_epochs = model_number_of_epochs()
    '''

    number_of_epochs = 20

    return number_of_epochs


def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    '''
    Purpose:
    ---
    All the required parameters for training are passed to this function.

    Input Arguments:
    ---
    1. model: An object of the 'Salary_Predictor' class
    2. number_of_epochs: For training the model
    3. tensors_and_iterable_training_data: list containing training and validation data tensors
                                             and iterable dataset object of training tensors
    4. loss_function: Loss function defined for the model
    5. optimizer: Optimizer defined for the model

    Returns:
    ---
    trained_model

    Example call:
    ---
    trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

    '''
    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in tensors_and_iterable_training_data[4]:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss}')
    trained_model = model
    return trained_model


def validation_function(trained_model, tensors_and_iterable_training_data):
    '''
    Purpose:
    ---
    This function will utilise the trained model to do predictions on the
    validation dataset. This will enable us to understand the accuracy of
    the model.

    Input Arguments:
    ---
    1. trained_model: Returned from the training function
    2. tensors_and_iterable_training_data: list containing training and validation data tensors
                                             and iterable dataset object of training tensors

    Returns:
    ---
    model_accuracy: Accuracy on the validation dataset

    Example call:
    ---
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

    '''
    trained_model.eval()
    X_val_tensor = tensors_and_iterable_training_data[1]
    y_val_tensor = tensors_and_iterable_training_data[3]

    with torch.no_grad():
        outputs = trained_model(X_val_tensor)
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y_val_tensor).float().mean()

    return accuracy.item()


def predict_salary():
    # Load the trained model
    trained_model = torch.jit.load('task_1a_trained_model.pth')

    # Switch the model to evaluation mode
    trained_model.eval()

    # # Get input from the user for feature values
    # print("Enter feature values for prediction:")
    # education = float(input("Education (0 for low, 1 for medium, 2 for high): "))
    # joining_year = float(input("Joining Year: "))
    # city = float(input("City (0 for CityA, 1 for CityB, etc.): "))
    # payment_tier = float(input("Payment Tier (0 for Tier1, 1 for Tier2, etc.): "))
    # age = float(input("Age: "))
    # gender = float(input("Gender (0 for Male, 1 for Female, etc.): "))
    # ever_benched = float(input("Ever Benched (0 for No, 1 for Yes): "))
    # experience_in_current_domain = float(input("Experience in Current Domain (in years): "))
    #
    # # Create a tensor from user input
    # input_tensor = torch.tensor(
    #     [[education, joining_year, city, payment_tier, age, gender, ever_benched, experience_in_current_domain]],
    #     dtype=torch.float32)
    #
    # # Make a prediction using the model
    # with torch.no_grad():
    #     prediction = trained_model(input_tensor)
    #     predicted_leave = "Leave" if prediction.item() > 0.5 else "Stay"
    #
    # # Display the prediction
    # print(f"Predicted: Employee will {predicted_leave}")


if __name__ == "__main__":
    task_1a_dataframe = pd.read_csv('task_1a_dataset.csv')

    encoded_dataframe = data_preprocessing(task_1a_dataframe)

    features_and_targets = identify_features_and_targets(encoded_dataframe)

    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)

    model = Salary_Predictor()

    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()

    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data,
                                      loss_function, optimizer)

model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)
print(f"Accuracy on the test set = {model_accuracy}")

X_train_tensor = tensors_and_iterable_training_data[0]
x = X_train_tensor[0]
jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")
predict_salary()
