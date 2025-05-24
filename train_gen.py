# train_model.py
from tqdm import tqdm
import torch
import os
import gc

def train_G(G,
            D,
            data_loader,
            loss_G,
            loss_D,
            optimizer_G,
            epochs,
            device,
            G_weight_path,
            cycle_num):

    G.to(device)
    D.to(device)
    
    G.train()
    D.eval()
    
    for epoch in range(epochs):
        G_list = []
        print("G: Cycle,Epoch: ",cycle_num, epoch)
        for data, label, _ in tqdm(data_loader):
            data = data.to(device)
            label = label.to(device)
            
            #predicted mask
            mask, _ = G(data)
            #score of discriminator
            score = D(mask)

            # train generator
            optimizer_G.zero_grad()
            
            G_accuracy_loss = loss_G(mask, label) 
            G_authen_loss = loss_D(score, torch.ones(score.shape).to(device)) 
            
            G_loss = G_accuracy_loss * 0.7 + G_authen_loss * 0.3 # trial 3
            
            G_loss.backward()
            optimizer_G.step()
            G_list.append(G_loss.cpu().detach().numpy())

        print("G loss: {:.6f}" .format(sum(G_list)/len(G_list)))
    if not os.path.exists(G_weight_path+f'cycle_{cycle_num}/'):
        os.makedirs(G_weight_path+f'cycle_{cycle_num}/')
    torch.save(G.state_dict(), G_weight_path+f'cycle_{cycle_num}/weights.pt') # save weight for each epoch