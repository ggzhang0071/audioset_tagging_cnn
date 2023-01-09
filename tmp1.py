import torch.profiler as profiler

import torchvision.models as models

model=models.resnet18()

with torch.profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    schedule=profiler.schedule(
        wait=2,
        warmup=2,
        active=6,
        repeat=1
    ),
    on_trace_ready=profiler.tensorboard_trace_handler("logs"),
    with_stack=True,
) as prof:
    for step, data in enumerate(train_loader):
        # forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()

        if step % 100 == 0:
            print("Step [{}/{}], Loss: {:.4f}"
                  .format(step, len(train_loader), loss.item()))
        
