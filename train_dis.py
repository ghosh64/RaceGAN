from tqdm import tqdm
import torch
import os
import gc

def train_D(G,
            D,
            data_loader,
            loss_D,
            optimizer_D,
            epochs,
            device,
            D_weight_path,
            cycle_num):

    G.to(device)
    D.to(device)
    
    G.eval()
    D.train()
    
    for epoch in range(epochs):
        D_list = []
        print("D: Cycle, Epoch: ", cycle_num, epoch)
        for data, label, _ in tqdm(data_loader):
            data = data.to(device)
            label = label.to(device)
            
            mask, _ = G(data)
            score = D(mask)
            
            real_mask_score = D(label)
            optimizer_D.zero_grad()
            
            fake_D_loss = loss_D(score, torch.zeros(mask.shape).to(device))
            real_D_loss = loss_D(real_mask_score, torch.ones(mask.shape).to(device))
            D_loss = fake_D_loss * 0.5 + real_D_loss * 0.5
            D_loss.backward()
            optimizer_D.step()
            D_list.append(D_loss.cpu().detach())

        print("D loss: {:.6f}" .format(sum(D_list)/len(D_list)))

    if not os.path.exists(D_weight_path+f'cycle_{cycle_num}/'):
        os.makedirs(D_weight_path+f'cycle_{cycle_num}/')

    torch.save(D.state_dict(), D_weight_path+f'cycle_{cycle_num}/weights.pt') # save weight for each epoch