# CIFAR10 Classification

## Описание
Репозиторий "**cifar10-classification**" содержит реализацию и сравнение двух нейросетевых архитектур (свёрточных) для задачи классификации изображений. Классы изображений: ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


## Цели
1. Сравнить работу двух свёрточных нейроынных сетей, в одной из которых есть Batch Normalization и Dropout, а в другой нет;
2. Реализовать кастомный вывод логов;
3. Реализовать метод, позволяющий посмотреть на промежуточные результаты работы свёрточных нейронных сетей;


## Стэк технологий
- Python
- PyTorch
- PyCharm, Jupyter Notebook


## Архитектура/Loss/Optimizer
1. Первая нейросеть:
   class NeuralNet_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    **Loss**: nn.CrossEntropyLoss()
    **Optimizer** = optim.SGD(net_v1.parameters(), lr=0.001, momentum=0.9)

2. Вторая нейросеть:
   # Архитектура второй сверточной нейросети
   class NeuralNet_v2(nn.Module):
      def __init__(self):
          super().__init__()
          self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
          self.pool = nn.MaxPool2d(2, 2) # (32, 16, 16)
          self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
          self.fc1 = nn.Linear(64 * 8 * 8, 120)
          self.fc2 = nn.Linear(120, 84)
          self.fc3 = nn.Linear(84, 10)
          self.bn1 = nn.BatchNorm2d(32)
          self.bn2 = nn.BatchNorm2d(64)
          self.dropout = nn.Dropout(0.3)
      def forward(self, x):
          x = self.pool(F.relu(self.bn1(self.conv1(x))))
          x = self.pool(F.relu(self.bn2(self.conv2(x))))
          x = torch.flatten(x, 1)
          x = self.dropout(F.relu(self.fc1(x)))
          x = self.dropout(F.relu(self.fc2(x)))
          x = self.fc3(x)
          return x

    **Loss**: nn.CrossEntropyLoss()
    **Optimizer** = optim.Adam(net_v2.parameters(), lr=0.001)


## Результаты обучения
| Архитектура | Количество эпох | Итоговый Loss | Время обучения | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NeuralNet_v1** | 60 | 0.1327 | 15.1 мин | 66.2% |
| **NeuralNet_v2** | 60 | 0.3610 | 17.4 мин | 73.2% |

   Важно: seed не был установлен, поэтому при повторении эксперимента могут получиться иные результаты.

