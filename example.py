import torch

import file_utils
import guided_transfer_learning as gtl

# --------------------------------------------------------------------------------
# --------------------- We first create a simple model ---------------------------
# --------------------------------------------------------------------------------


class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(2, 15)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(15, 5)
        self.linear3 = torch.nn.Linear(5, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tiny_model = TinyModel().to(device)
loss_function = torch.nn.L1Loss()
optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.05)


def train_model(model, x, y, epochs, guidance_matrix=None):
    for _ in range(epochs):
        result = model(x)
        loss = loss_function(result, y)  # torch.tensor([1.00, 2.00], dtype=torch.float)
        print("loss", loss)
        model.zero_grad()
        loss.backward()
        if guidance_matrix is not None:
            gtl.apply_guidance(model, guidance_matrix)
        optimizer.step()


# --------------------------------------------------------------------------------
# --------------------- Now we use  the GTL library ------------------------------
# --------------------------------------------------------------------------------

# A handy GTL function to list all the model parameter names and the sizes of each
param_list = gtl.list_params(tiny_model)
print(param_list)

# Create some data for training the model
base_train_X = torch.tensor([[0.1, 0.2], [0.11, 0.22], [0.3, 0.4], [0.3, 0.4]]).to(
    device
)
base_train_Y = torch.tensor([[1.1, 2.2], [1.1, 2.2], [1.3, 2.4], [0.3, 0.4]]).to(device)

# Another handy function to print the few values of each parameter name to monitor changes
print(gtl.get_param_values(tiny_model, 3))

# Train the base model
print("Training base model")
train_model(model=tiny_model, x=base_train_X, y=base_train_Y, epochs=10)

# Use GTL library to save the base model if we think that we may need it later
file_utils.save_base(tiny_model)
print("base saved")

# delete the model so that we can load it again.
del tiny_model

# Now create new model and load parameters.
# This simulates that situation in which the base model has been trained previously.
tiny_model = TinyModel().to(device)
file_utils.load_base(tiny_model)

# Example of the complex scout indexes
# A = [[1,2,4],
#      [[7],[9]],
#      [3,5,6],
#      [[5],[20], 23, 24, [28],[30]],
#      [40, 45, 46]]
scout_indexes = [[0, 1], [2, 3]]

scout_data_X = gtl.create_scout_data_from_ranged_indexes(scout_indexes, base_train_X)
scout_data_Y = gtl.create_scout_data_from_ranged_indexes(scout_indexes, base_train_Y)
print("There are ", len(scout_data_X), "scouts")
print(scout_data_X)

number_of_scouts = len(scout_data_X)

print("Training ", number_of_scouts, "scouts")
my_scouts = gtl.create_scouts(tiny_model)
for s in range(number_of_scouts):
    print("scout number:", s)
    file_utils.load_base(tiny_model)
    train_model(model=tiny_model, x=scout_data_X[s], y=scout_data_Y[s], epochs=10)
    my_scouts.add_scout(tiny_model)
print("A total of", len(my_scouts), "trained")

guidance_matrix = my_scouts.create_raw_guidance(device)
del my_scouts

# A handy GTL function to print some of the values in guidance matrix
print("---------------------------------------------------------------")
print("---------------------------------------------------------------")
print(gtl.get_guidance_values(guidance_matrix, 5))

# A GTL function to plot the distribution of values in the guidance matrix
# gtl.plot_guidance(guiding_matrix, name="linear1.weight")
print("---------------------------------------------------------------")
print("guide matrix device: ", guidance_matrix["linear1.weight"].get_device())
print(guidance_matrix)

gtl.adjust_guidance_matrix(guidance_matrix, "zero_enforced_and_normalized", "room")
print("---------------------------------------------------------------")
print(guidance_matrix)
gtl.adjust_guidance_matrix(
    guidance_matrix, gtl.Focus.ZERO_ENFORCED_AND_NORMALIZED, gtl.Temp.CHILLY
)
gtl.adjust_guidance_matrix(
    guidance_matrix, gtl.Focus.ZERO_ENFORCED_AND_NORMALIZED, gtl.Temp.FREEZING
)
gtl.adjust_guidance_matrix(
    guidance_matrix, gtl.Focus.ZERO_ENFORCED_AND_NORMALIZED, gtl.Temp.WARM
)
gtl.adjust_guidance_matrix(
    guidance_matrix, gtl.Focus.ZERO_ENFORCED_AND_NORMALIZED, gtl.Temp.EVAPORATING
)
gtl.adjust_guidance_matrix(
    guidance_matrix, gtl.Focus.ZERO_ENFORCED_AND_NORMALIZED, gtl.Temp.ICY
)

# --------------------------------------------------------------------------------
# --------------------- Finally performing guided transfer learning --------------
# --------------------------------------------------------------------------------

print("Transfer learning with guidance matrix")
train_model(
    model=tiny_model,
    x=base_train_X,
    y=base_train_Y,
    epochs=10,
    guidance_matrix=guidance_matrix,
)
