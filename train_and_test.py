import torch
import torch.nn as nn
import torch.nn.functional as F 


def valid(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        loss = []
        accuracy = []
        iteration =0
        a = [0]*4
        for data_sample in test_loader:
            y = data_sample['label'].to(device)
            outputs = model(data_sample['ligase_atom'].to(device),
                            data_sample['ligase_bond'].to(device),
                            data_sample['target_atom'].to(device),
                            data_sample['target_bond'].to(device),
                            data_sample['ligase_ligand_atom'].to(device),
                            data_sample['ligase_ligand_bond'].to(device),
                            data_sample['target_ligand_atom'].to(device),
                            data_sample['target_ligand_bond'].to(device),
                            data_sample['smiles'].to(device))
            criterion = nn.CrossEntropyLoss()
            loss_val = criterion(outputs, y)
            pred_y = torch.max(outputs,1)[1].cpu().numpy()
            y = y.cpu().numpy()
            if y[0] == 1 and pred_y[0] == 1:
                a[3] +=1
            elif y[0] == 1 and pred_y[0] == 0:
                a[2] +=1
            elif y[0] == 0 and pred_y[0] == 1:
                a[1] +=1
            elif y[0] == 0 and pred_y[0] == 0:
                a[0] +=1
            accuracy.append(float((pred_y == y).astype(int).sum()) / float(y.size))
            loss.append(loss_val.item())
            iteration += 1
        model.train()
        print(a)
    return sum(loss)/iteration, sum(accuracy)/iteration

def train(model, lr, epoch, train_loader, valid_loader, device, writer, LOSS_NAME, VAL_INTERVAL=2):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    running_loss = 0.0
    best_val_acc = 0.0

    for epo in range(epoch):
        total_num = 0
        for batch_idx, data_sample in enumerate(train_loader):
            outputs = model(data_sample['ligase_atom'].to(device),
                            data_sample['ligase_bond'].to(device),
                            data_sample['target_atom'].to(device),
                            data_sample['target_bond'].to(device),
                            data_sample['ligase_ligand_atom'].to(device),
                            data_sample['ligase_ligand_bond'].to(device),
                            data_sample['target_ligand_atom'].to(device),
                            data_sample['target_ligand_bond'].to(device),
                            data_sample['smiles'].to(device))
            batch_size, _ = data_sample["ligase_atom"].shape
            total_num += batch_size
            criterion = nn.CrossEntropyLoss()
            y = data_sample['label'].to(device)
            loss = criterion(outputs, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
        writer.add_scalar(LOSS_NAME, running_loss / total_num, epo)
        print('Train epoch %d, loss: %.4f' % (epo, running_loss))
        running_loss =0.0
        if (epo+1) % 50 == 0:
            val_loss, val_acc = valid(model, valid_loader, device)
            print(f'Valid epoch {epo}, loss: {val_loss:.4f}, acc: {val_acc}')
            # torch.save(model, 'model/%s_%d'%(LOSS_NAME, epo+1))
    return model