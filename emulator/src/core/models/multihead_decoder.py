import torch
import torch.nn as nn

class MultiHeadDecoder(nn.Module):
    def __init__(self, in_var_ids, out_var_ids, train_models, test_models, n_layers, hidden_dim):
        super().__init__()
        self.num_input_vars = len(in_var_ids)
        self.num_output_vars = len(out_var_ids)
        if test_models is None:
            test_models=[]
        total_models = list(set(train_models+test_models))
        self.n_heads = len(total_models)
        print("Setting up decoder for the following models:", total_models)
        print(f"{self.n_heads} total heads")
        model_name_to_head_num = dict()
        for i,m in enumerate(total_models):
            model_name_to_head_num[m]=i
        print(model_name_to_head_num)
        self.model_name_to_head_num = model_name_to_head_num

        self.hidden_dim = hidden_dim
        self.heads = nn.ModuleList()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        for _ in range(self.n_heads):
            layers = []
            layers.append(nn.Conv2d(self.num_output_vars, self.hidden_dim, kernel_size=1))
            for _ in range(n_layers - 1):
                layers.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1))
            layers.append(nn.Conv2d(self.hidden_dim, self.num_output_vars, kernel_size=1))
            head = nn.Sequential(*layers)
           
            self.heads.append(head)
        
        self.heads.to(self.device)
        
        
    
    """
    def forward(self, x, head_num):
        # assumes single head_num
        self.set_active_head(head_num)
        head = self.heads[head_num]
        out = head(x)
        return out
    """

    def forward(self, x, model_ids):
        # model ids (str) may be multiple (batch_size, 1)

        # convert model ids to head_nums
        head_nums = torch.tensor([self.model_name_to_head_num[id] for id in model_ids])

        # head_nums (batch_size, 1)
        unique_heads = torch.unique(head_nums)
        self.set_active_heads(unique_heads)

        # creaty empty tensor of output shape
        y = torch.empty(x.shape[0], x.shape[1], self.num_output_vars, x.shape[-2], x.shape[-1]).to(self.device)
        # fill in values by uniqe head
        
        for i,h in enumerate(head_nums):
            y[i,:]=self.heads[h](x[i,:])
        
        return y


    def set_active_heads(self, head_nums):
        # head_nums list of head_nums
        for i,head in enumerate(self.heads):
            for param in head.parameters():
                if i in head_nums:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
    
    def set_active_head(self, head_num): # single head_num
        # Freeze all parameters
        for head in self.heads:
            for param in head.parameters():
                param.requires_grad_(False)

        # Unfreeze parameters of the active head
        for param in self.heads[head_num].parameters():
            param.requires_grad_(True)


if __name__=='__main__':
        md = MultiHeadDecoder(in_var_ids=['BC', 'CO2'], out_var_ids=['pr'], train_models=['model1', 'model2', 'model3'], n_layers=2, hidden_dim=24)
        x = torch.ones((4,12,2,64,64))
        num_idx = torch.as_tensor([0,0,0,0])

        y = md.forward_multi_head(x,num_idx)
        print(y.shape)