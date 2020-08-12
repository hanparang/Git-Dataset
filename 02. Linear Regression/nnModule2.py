import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3,1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epoch = 2000
for epoch in range(nb_epoch+1):
    H = model(x_train)

    cost = F.mse_loss(H, y_train)

    optimizer.zero_grad()

    cost.backward()

    optimizer.step()

    if epoch % 100 == 0:
        # 100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epoch, cost.item()
      ))

new_var = torch.FloatTensor([[73,80,75]])
pred_y = model(new_var)

print("new_var pred_y = ",pred_y)