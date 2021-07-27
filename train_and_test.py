import torch
import torch.nn as nn
import torch.nn.functional as F 

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()

        self.smoothing = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == "mean" else loss.sum()

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        if self.weight is not None:
            self.weight = self.weight.to(self.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight)

        return self.linear_combination(loss / n, nll)


def valid(model, test_loader, device):
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

def train(model, lr, epoch, train_loader, valid_loader, device, writer, LOSS_NAME, VAL_INTERVAL):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    running_loss = 0.0
    best_val_acc = 0.0
    # train_loss = LabelSmoothingLoss(smoothing=0.1)

    for epo in range(epoch):
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
            criterion = nn.CrossEntropyLoss()
            y = data_sample['label'].to(device)
            loss = criterion(outputs, y)
            # loss = train_loss(outputs, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # if ((batch_idx + 1) % accum_step == 0) or (batch_idx + 1 == len(train_loader)): 
            #     opt.step()
            #     opt.zero_grad()
            running_loss += loss.item()
        writer.add_scalar(LOSS_NAME, running_loss, epo)
        print('Train epoch %d, loss: %.4f' % (epo, running_loss))
        running_loss =0.0
        if (epo+1) % VAL_INTERVAL == 0:
            val_loss, val_acc = valid(model, valid_loader, device)
            print(f'Valid epoch {epo}, loss: {val_loss:.4f}, acc: {val_acc}')
            writer.add_scalar(LOSS_NAME + "_valid", val_loss, epo)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(model, LOSS_NAME + "_best.pth")
    return model