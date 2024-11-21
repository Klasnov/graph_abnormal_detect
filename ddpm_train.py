import os
import time
import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_NODE = 100
TYPES = ["THREAD", "FILE", "REGISTRY", "FLOW", "USER_SESSION", "SERVICE", "PROCESS", "MODULE", "TASK", "SHELL"]
ACTIONS = [
    "FILE_CREATE", "FILE_DELETE", "FILE_MODIFY", "FILE_READ", "FILE_RENAME", "FILE_WRITE", "FLOW_MESSAGE",
    "FLOW_OPEN", "MODULE_LOAD", "PROCESS_CREATE", "PROCESS_OPEN", "PROCESS_TERMINATE", "REGISTRY_ADD",
    "REGISTRY_EDIT", "REGISTRY_REMOVE", "SERVICE_CREATE", "SHELL_COMMAND", "TASK_CREATE", "TASK_DELETE",
    "TASK_MODIFY", "TASK_START", "THREAD_CREATE", "THREAD_REMOTE_CREATE", "THREAD_TERMINATE", "USER_SESSION_GRANT",
    "USER_SESSION_INTERACTIVE","USER_SESSION_LOGIN", "USER_SESSION_LOGOUT", "USER_SESSION_REMOTE", "USER_SESSION_UNLOCK"
]
NUM_BATCH = 1768
NUM_GRAPH = NUM_BATCH * BATCH_SIZE
NUM_WARMUP = 2 * max(NUM_NODE, NUM_GRAPH // BATCH_SIZE)

EPOCH_NUM = 1000
INIT_LR = 0.0003


def get_weighted_adjacency_matrix(graph):
    adj_spr = graph.adjacency_matrix(scipy_fmt="coo")
    edge_freq = graph.edata["action"]
    num_nodes = graph.number_of_nodes()
    adj = torch.zeros(num_nodes, num_nodes)
    for i, j, freq in zip(adj_spr.row, adj_spr.col, edge_freq):
        adj[i, j] = freq
    adj = adj + adj.T
    return adj


def get_batch(path="data/train/", index=0):
    save_path = path + "batch_" + str(index) + "/"
    if not os.path.exists(save_path):
        raise ValueError("The path does not exist")
    h = torch.load(save_path + "h_" + str(index) + ".pt", weights_only=True)
    pe = torch.load(save_path + "pe_" + str(index) + ".pt", weights_only=True)
    e = torch.load(save_path + "e_" + str(index) + ".pt", weights_only=True)
    return h, pe, e


def sym_tensor(x):
    x = x.permute(0, 3, 1, 2) # [bs, n, n, d]
    triu = torch.triu(x,diagonal=1).transpose(3,2) # [bs, d, n, n]
    mask = (triu.abs()>0).float()                  # [bs, d, n, n]
    x =  x * (1 - mask ) + mask * triu             # [bs, d, n, n]
    x = x.permute(0, 2, 3, 1) # [bs, n, n, d]
    return x               # [bs, n, n, d]

class Embed_G(nn.Module):
    def __init__(self, params):
        super().__init__()
        d = params["d"]
        self.Embed_h = nn.Embedding(params["num_type"], d)
        self.Embed_e = nn.Embedding(params["num_action"], d)
        self.Embed_pe = nn.Embedding(params["num_node"], d)
    
    def forward(self, h, e, pe):
        pe = self.Embed_pe(pe) # [bs, n, d]
        h = self.Embed_h(h)
        h = h + pe
        e = self.Embed_e(e) # [bs, n, n, d]
        e = e + pe.unsqueeze(1) + pe.unsqueeze(2)
        e = sym_tensor(e)
        return h, e


class Attention_Layer(nn.Module):
    def __init__(self, d, d_head, drop):
        super().__init__()
        self.Q = nn.Linear(d, d_head)
        self.K = nn.Linear(d, d_head)
        self.V = nn.Linear(d, d_head)
        self.E = nn.Linear(d, d_head)
        self.Ni = nn.Linear(d, d_head)
        self.Nj = nn.Linear(d, d_head)
        self.Drop_Att = nn.Dropout(drop)
        self.sqrt_d = torch.sqrt(torch.tensor(d_head))
    
    def forward(self, h, e):
        # h: [bs, n, d]; e: [bs, n, n, d]
        Q = self.Q(h) # [bs, n, d_head]
        K = self.K(h)
        V = self.V(h)
        Q = Q.unsqueeze(2)  # [bs, n, 1, d_head]
        K = K.unsqueeze(1)  # [bs, 1, n, d_head]
        E = self.E(e)       # [bs, n, n, d_head]
        Ni = self.Ni(h).unsqueeze(2) # [bs, n, 1, d_head]
        Nj = self.Nj(h).unsqueeze(1) # [bs, 1, n, d_head]
        e = E + Ni + Nj              # [bs, n, n, d_head]
        Att = (Q * e * K).sum(dim=-1) / self.sqrt_d # [bs, n, n]
        Att = torch.softmax(Att, dim=1)             # [bs, n, n]
        Att = self.Drop_Att(Att)
        h = Att @ V # [bs, n, d_head]
        return h, e


class MAH_Layer(nn.Module):
    def __init__(self, d, head_num, drop):
        super().__init__()
        d_head = d // head_num
        self.heads = nn.ModuleList([Attention_Layer(d, d_head, drop) for _ in range(head_num)])
        self.WO_h = nn.Linear(d, d)
        self.WO_e = nn.Linear(d, d)
        self.Drop_h = nn.Dropout(drop)
        self.Drop_e = nn.Dropout(drop)
    
    def forward(self, h, e):
        # h: [bs, n, d]; e: [bs, n, n, d]
        h_MHA = []
        e_MHA = []
        for head in self.heads:
            h_mha, e_mha = head(h, e)
            h_MHA.append(h_mha)
            e_MHA.append(e_mha)
        h = self.Drop_h(self.WO_h(torch.cat(h_MHA, dim=2)))
        e = self.Drop_e(self.WO_e(torch.cat(e_MHA, dim=3)))
        return h, e


class GT_Layer(nn.Module):
    def __init__(self, d, num_head, drop):
        super().__init__()
        self.Norm_h_1 = nn.LayerNorm(d)
        self.Norm_e_1 = nn.LayerNorm(d)
        self.MHA = MAH_Layer(d, num_head, drop)
        self.Norm_h_2 = nn.LayerNorm(d)
        self.Norm_e_2 = nn.LayerNorm(d)
        self.MLP_h = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))
        self.MLP_e = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))
        self.Drop_h = nn.Dropout(drop)
        self.Drop_e = nn.Dropout(drop)
    
    def forward(self, h, e):
        # h: [bs, n, d]; e: [bs, n, n, d]
        h = self.Norm_h_1(h)
        e = self.Norm_e_1(e)
        h_MHA, e_MHA = self.MHA(h, e)
        h = h + h_MHA
        h = h + self.MLP_h(self.Norm_h_2(h))
        e = e + e_MHA
        e = e + self.MLP_e(self.Norm_e_2(e))
        h = self.Drop_h(h)
        e = self.Drop_e(e)
        return h, e


class UNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        d = params["d"]
        # Graph Embedding
        self.num_node = params["num_node"]
        self.num_t = params["num_t"]
        self.num_type = params["num_type"]
        self.num_action = params["num_action"]
        self.Embed_h = nn.Linear(self.num_type, d)
        self.Embed_e = nn.Linear(self.num_action, d)
        self.pe_h = nn.Embedding(self.num_node, d)
        # Time Embedding
        self.pe_t = nn.Sequential(nn.Embedding(self.num_t, d), nn.ReLU(), nn.Linear(d, d))
        # GT Layers
        num_gt_layer = params["num_gt_layer"]
        num_head = params["num_head"]
        drop = params["drop"]
        self.GT_Layers = nn.ModuleList([GT_Layer(d, num_head, drop) for _ in range(num_gt_layer)])
        # Output Layer
        self.LN_h = nn.Linear(d, self.num_type)
        self.LN_e = nn.Linear(d, self.num_action)
    
    def forward(self, h, e, pe_h, sample_t):
        # Embedding for Graph
        pe_h = self.pe_h(pe_h) # [bs, n, d]
        h_t = self.Embed_h(h)
        h_t = h_t + pe_h
        e_t = self.Embed_e(e) # [bs, n, n, d]
        e_t = e_t + pe_h.unsqueeze(1)
        e_t = sym_tensor(e_t)
        # Embedding for Time
        pe_t = self.pe_t(sample_t) # [bs, d]
        # GT Layers
        for GT_Layer in self.GT_Layers:
            h_t = h_t + pe_t.unsqueeze(1) # [bs, n, d]
            e_t = e_t + pe_t.unsqueeze(1).unsqueeze(2) # [bs, n, n, d]
            h_t, e_t = GT_Layer(h_t, e_t)
            e_t = sym_tensor(e_t)
        # Output Layer
        h_t_minus_one = self.LN_h(h_t)
        e_t_minus_one = self.LN_e(e_t)
        return h_t_minus_one, e_t_minus_one
    

class DDPM(nn.Module):
    def __init__(self, num_t, beta_1, beta_t, params):
        super().__init__()
        self.device = params["device"]
        self.num_type = params["num_type"]
        self.num_action = params["num_action"]
        self.UNet = UNet(params)
        self.num_t = num_t
        self.alpha_t = 1.0 - torch.linspace(beta_1, beta_t, num_t).to(self.device)
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0)
    
    def forward(self, h_0, e_0, sample_t, noise_h0, noise_e0):
        h0 = torch.nn.functional.one_hot(h_0, self.num_type).float()
        e0 = torch.nn.functional.one_hot(e_0, self.num_action).float()
        bs = len(sample_t)
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar_t[sample_t])
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - self.alpha_bar_t[sample_t])
        h_t = sqrt_alpha_bar_t.view(bs, 1, 1) * h0 + sqrt_one_minus_alpha_bar_t.view(bs, 1, 1) * noise_h0
        e_t = sqrt_alpha_bar_t.view(bs, 1, 1, 1) * e0 + sqrt_one_minus_alpha_bar_t.view(bs, 1, 1, 1) * noise_e0
        return h_t, e_t
    
    def backward(self, h_t, e_t, pe_h, sample_t):
        noise_pred_h_t, noise_pred_e_t = self.UNet(h_t, e_t, pe_h, sample_t)
        return noise_pred_h_t, noise_pred_e_t
    

def train_ddpm(num_t, beta_1, beta_t, net_params, load_save=True, model_path="model/"):
    torch.random.manual_seed(0)
    ddpm = DDPM(num_t, beta_1, beta_t, net_params).to(DEVICE)
    if load_save:
        if os.path.exists(model_path + "ddpm.pt"):
            ddpm.load_state_dict(torch.load(model_path + "ddpm.pt", weights_only=True))
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    init_lr = INIT_LR
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=init_lr)
    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: min((t+1)/NUM_WARMUP, 1.0))
    scheduler_tracker = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=1)

    num_epoch = EPOCH_NUM
    num_warmup_batch = 0

    train_loss_drop_patience = 5
    loss_dropping = True
    train_loss_drop_cnt = 0
    previous_best_loss = float("inf")

    start = time.time()
    for epoch in range(num_epoch):
        running_loss = 0.0
        num_batch = 0

        ddpm.train()

        for i in range(NUM_BATCH):
            h, pe, e = get_batch(index=i)
            h = h.to(DEVICE)
            pe = pe.to(DEVICE)
            e = e.to(DEVICE)
            batch_sample_t = torch.randint(0, num_t, (BATCH_SIZE,)).long().to(DEVICE)
            batch_noise_h_t = torch.randn(BATCH_SIZE, NUM_NODE, len(TYPES)).to(DEVICE)
            batch_noise_e_t = torch.randn(BATCH_SIZE, NUM_NODE, NUM_NODE, len(ACTIONS)).to(DEVICE)
            batch_noise_e_t = sym_tensor(batch_noise_e_t)

            h_t, e_t = ddpm(h, e, batch_sample_t, batch_noise_h_t, batch_noise_e_t)
            noise_pred_h_t, noise_pred_e_t = ddpm.backward(h_t, e_t, pe, batch_sample_t)

            loss = torch.nn.MSELoss()(noise_pred_h_t, batch_noise_h_t) + torch.nn.MSELoss()(noise_pred_e_t, batch_noise_e_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if num_batch < NUM_WARMUP:
                scheduler_warmup.step()
            num_batch += 1

            running_loss += loss.detach().item()
            num_batch += 1

            del h, pe, e, batch_sample_t, batch_noise_h_t, batch_noise_e_t, h_t, e_t, noise_pred_h_t, noise_pred_e_t
            torch.cuda.empty_cache()
        
        mean_loss = running_loss / num_batch
        if num_warmup_batch >= NUM_WARMUP:
            scheduler_tracker.step(mean_loss)
        elapsed = (time.time() - start) / 60
        print(f"Epoch {epoch+1}/{num_epoch}  Loss: {mean_loss:.6f}  lr: {optimizer.param_groups[0]['lr']:.6f}  Time: {elapsed:.2f} mins")

        if optimizer.param_groups[0]['lr'] < 1e-5:
            print("Early stopping since lr < 1e-5")
            break

        if mean_loss < previous_best_loss:
            previous_best_loss = mean_loss
            train_loss_drop_cnt = 0
            loss_dropping = True
        else:
            train_loss_drop_cnt += 1
            loss_dropping = False
            if train_loss_drop_cnt >= train_loss_drop_patience:
                print("Early stopping since loss is not dropping")
                break

        if (epoch + 1) % 5 == 0 and loss_dropping:
            torch.save(ddpm.state_dict(), model_path + "ddpm.pt")
    
    torch.save(ddpm.state_dict(), model_path + "ddpm.pt")
    print(f"Finished the training of DDPM, with the best loss {previous_best_loss:.6f}, and the total time {elapsed:.2f} mins")


if __name__ == "__main__":
    beta_1 = 0.0001
    beta_t = 0.1
    num_t = 200

    ddpm_net_params = {
        "num_type": len(TYPES),
        "num_action": len(ACTIONS),
        "num_node": NUM_NODE,
        "num_gt_layer": 6,
        "num_head": 4,
        "d": 32 * 4,
        "num_t": num_t,
        "drop": 0,
        "device": DEVICE
    }

    train_ddpm(num_t, beta_1, beta_t, ddpm_net_params)
