# Pytorch template

Personal pytorch template

## Automatic Mixed Precision (Apex+pytorch)

#### Requirements

- pytorch 1.5 or older
- apex

```
1. import amp module
from apex import amp

2. use amp to initialize model and optimizer 
net, optimizer = amp.initialize(net, optimizer, opt_level='O1')

3. use scaled_loss.backward() instead of normal loss.backward()
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

## Automatic Mixed Precision (pytorch1.6 native AMP)

#### Requirements

- pytorch 1.6

```
1. import amp module
from torch.cuda.amp import GradScaler, autocast

2. declear scaler
scaler = GradScaler()

3. autocast and gradscaler
with autocast():
    outputs = net(inputs)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

TODO:
- [x] train template
- [ ] evaluate template
- [ ] use .json as config file
- [ ] tensorboard
- [ ] Distributed Data Parallel
- [x] Automatic Mixed Precision (Apex+pytorch)
- [x] Automatic Mixed Precision (pytorch1.6 native AMP)
