import torch
from torch.utils.data import TensorDataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def update_model(
    model,
    mll,
    train_z,
    train_y,
    n_epochs=30,
    learning_rte=0.001,
    grad_clip=2.0,
    bsz=128,
):
    model.train() 
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rte} ], lr=learning_rte)
    train_bsz = min(len(train_y),bsz)
    train_dataset = TensorDataset(train_z.to(device), train_y.to(device))
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in range(n_epochs):
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.to(device))
            loss = -mll(output, scores.to(device))
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
    model.eval()

    return model
