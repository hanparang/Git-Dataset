import torch
import torch.nn.functional as F
import torch.nn as nn

torch.manual_seed(1)
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(1, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 2000

for epoch in range(nb_epochs+1):
    h = model(x_train)

    cost = F.mse_loss(h, y_train)

    optimizer.zero_grad()

    cost.backward()

    optimizer.step()

    if epoch % 100 == 0:
        # 100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
      ))

new_var = torch.FloatTensor([[4.0]])
pred_y = model(new_var)
print(pred_y)