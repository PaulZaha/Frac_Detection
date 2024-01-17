import numpy as np

Training_val = [0.5284, 0.3712, 0.3164, 0.2798, 0.2503, 0.2247, 0.2091, 0.2046, 0.1884, 0.1824, 0.1716, 0.1610, 0.1536, 0.1512, 0.1401, 0.1372, 0.1348, 0.1293, 0.1276, 0.1209]
Validation_val = [0.4294, 0.3548, 0.3197, 0.2712, 0.2669, 0.2485, 0.2319, 0.2261, 0.2248, 0.2171, 0.2091, 0.2051, 0.2021, 0.1996, 0.1919, 0.1850, 0.1846, 0.1749, 0.1767, 0.1827]


initial_rate = 1e-5

epochs = 20

train_avg = []
val_avg = []

for epoch in range(2,epochs):
    train_avg.append(round(((Training_val[epoch-2]+Training_val[epoch-1]+Training_val[epoch])/3),4))
    
    val_avg.append(round(((Validation_val[epoch-2]+Validation_val[epoch-1]+Validation_val[epoch])/3),4))


residual_avg = []
for i in range(18):
    residual_avg.append(round((val_avg[i]-train_avg[i]),4))

print(residual_avg)

decay = []
for i in range(18):
    decay.append(initial_rate-(residual_avg[i]*initial_rate))

lr_list = []
for i in range(10):
    lr_list.append(initial_rate)

for i in range(10):
    lr_list.append(decay[i+8])

print(lr_list)