import torch
import torch.nn as nn
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_NODE = 100
TYPES = ["THREAD", "FILE", "REGISTRY", "FLOW", "USER_SESSION", "SERVICE", "PROCESS", "MODULE", "TASK", "SHELL"]
ACTIONS = [
    "FILE_CREATE", "FILE_DELETE", "FILE_MODIFY", "FILE_READ", "FILE_RENAME", "FILE_WRITE", "FLOW_MESSAGE",
    "FLOW_OPEN", "MODULE_LOAD", "PROCESS_CREATE", "PROCESS_OPEN", "PROCESS_TERMINATE", "REGISTRY_ADD",
    "REGISTRY_EDIT", "REGISTRY_REMOVE", "SERVICE_CREATE", "SHELL_COMMAND", "TASK_CREATE", "TASK_DELETE",
    "TASK_MODIFY", "TASK_START", "THREAD_CREATE", "THREAD_REMOTE_CREATE", "THREAD_TERMINATE", "USER_SESSION_GRANT",
    "USER_SESSION_INTERACTIVE","USER_SESSION_LOGIN", "USER_SESSION_LOGOUT", "USER_SESSION_REMOTE", "USER_SESSION_UNLOCK"
]
BATCH_SIZE = 32
MAX_BATCH = 1768
NUM_BATCH = 177

torch.random.manual_seed(0)
rand_indices = torch.randperm(MAX_BATCH)
pick_indices = rand_indices[:NUM_BATCH]
pick_indices = pick_indices.tolist()


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


class VAE(nn.Module):
    def __init__(self, params):
        super().__init__()
        d = params["d"]
        
        # Graph Embedding
        self.num_node = params["num_node"]
        self.Embed_he = Embed_G(params)
        self.Embed_pe = nn.Embedding(params["num_node"], d)

        # GT Layers
        num_enc_layer = params["num_enc_layer"]
        num_dec_layer = params["num_dec_layer"]
        num_head = params["num_head"]
        drop = params["drop"]
        self.Enc_Layers = nn.ModuleList([GT_Layer(d, num_head, drop) for _ in range(num_enc_layer)])
        self.Dec_Layers = nn.ModuleList([GT_Layer(d, num_head, drop) for _ in range(num_dec_layer)])

        # Encoder
        dz = params["dz"]
        self.LN_q_mu = nn.Linear(d, dz)
        self.LN_q_logvar = nn.Linear(d, dz)

        # Decoder
        self.LN_p = nn.Linear(dz, d)

        # Output Layer
        self.Norm_Out_h = nn.LayerNorm(d)
        self.Norm_Out_e = nn.LayerNorm(d)
        self.LN_h = nn.Linear(d, params["num_type"])
        self.LN_e = nn.Linear(d, params["num_action"])
    
    def forward(self, h, e, pe, num_node=None):
        if num_node is None:
            num_node = self.num_node

        # Embedding
        h, e = self.Embed_he(h, e, pe)
        n = h.size(1)
        pe = self.Embed_pe(pe)
        # Encoder
        for Enc_Layer in self.Enc_Layers:
            h, e = Enc_Layer(h, e)
            e = sym_tensor(e)
        graph_token = h.mean(dim=1)
        q_mu = self.LN_q_mu(graph_token)
        q_logvar = self.LN_q_logvar(graph_token)
        q_std = torch.exp(q_logvar / 2)
        eps = torch.randn_like(q_std)
        z = q_mu + eps * q_std # [bs, dz]
        n = h.size(1)

        # Decoder
        z = self.LN_p(z) # [bs, d]
        h = z.unsqueeze(1).repeat(1, n, 1) # [bs, n, d]
        h = h + pe
        e = z.unsqueeze(1).unsqueeze(1).repeat(1, n, n, 1) # [bs, n, n, d]
        e = e + pe.unsqueeze(1) + pe.unsqueeze(2)
        e = sym_tensor(e)
        for Dec_Layer in self.Dec_Layers:
            h, e = Dec_Layer(h, e)
            e = sym_tensor(e)
        h = self.Norm_Out_h(h)
        e = self.Norm_Out_e(e)
        h = self.LN_h(h)
        e = self.LN_e(e)
        return h, e, q_mu, q_logvar
    

if __name__ == "__main__":
    vae_net_params = {
        "num_type": len(TYPES),
        "num_action": len(ACTIONS),
        "num_node": NUM_NODE,
        "num_enc_layer": 4,
        "num_dec_layer": 4,
        "num_head": 8,
        "drop": 0,
        "d": 16 * 8,
        "dz": 32
    }

    vae = vae = VAE(vae_net_params).to(DEVICE)
    vae.load_state_dict(torch.load("model/vae.pt", weights_only=True))

    loss_list = []
    for i in range(NUM_BATCH):
        h, pe, e = get_batch(index=pick_indices[i])
        h = h.to(DEVICE)
        pe = pe.to(DEVICE)
        e = e.to(DEVICE)
        pred_h, pred_e, q_mu, q_logvar = vae(h, e, pe)
        loss_data = torch.nn.CrossEntropyLoss()(pred_h.view(BATCH_SIZE * NUM_NODE, len(TYPES)), h.view(BATCH_SIZE * NUM_NODE)) 
        loss_data += torch.nn.CrossEntropyLoss()(pred_e.view(BATCH_SIZE * NUM_NODE * NUM_NODE, len(ACTIONS)), e.view(BATCH_SIZE * NUM_NODE * NUM_NODE))
        loss_kl = -0.5 * torch.sum(1 + q_logvar - q_mu.pow(2) - q_logvar.exp())
        loss = 2.5 * loss_data + loss_kl
        loss_list.append(loss.detach().item())

        del h, pe, e, pred_h, pred_e, q_mu, q_logvar, loss_data, loss_kl, loss
        torch.cuda.empty_cache()


    loss = torch.tensor(loss_list)
    if not os.path.exists("results/"):
        os.makedirs("results/")
    torch.save(loss, "results/vae_normal.pt")
    print(f"Mean Loss: {loss.mean().item():.6f}, Std Loss: {loss.std().item():.6f}")
