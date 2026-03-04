def bpr_loss(self, user_inputs, pos_item_inputs, neg_item_inputs, type="user"):

    user_emb, item_emb, group_emb = self.compute_embeddings()

    pos_i_emb = item_emb[pos_item_inputs]
    neg_i_emb = item_emb[neg_item_inputs]

    if type == "group":
        member_list = self.get_member_list(user_inputs)
        member_masked, _, _ = self.get_member_mask(user_inputs, member_list)

        influ_emb = self.get_influence_embedding(
            member_masked, user_inputs, user_emb
        )

        pos_scores = self.compute_score(influ_emb, pos_i_emb)
        neg_scores = self.compute_score(influ_emb, neg_i_emb)

    else:
        u_emb = user_emb[user_inputs]

        pos_scores = self.compute_score(u_emb, pos_i_emb)
        neg_scores = self.compute_score(u_emb, neg_i_emb)

    return torch.mean(F.softplus(neg_scores - pos_scores))

def forward(self, user_inputs, item_inputs, type="user"):

    user_emb, item_emb, group_emb = self.compute_embeddings()

    i_emb = item_emb[item_inputs]

    if type == "group":
        member_list = self.get_member_list(user_inputs)
        member_masked, _, _ = self.get_member_mask(user_inputs, member_list)

        influ_emb = self.get_influence_embedding(
            member_masked, user_inputs, user_emb
        )

        scores = self.compute_score(influ_emb, i_emb)
        return scores

    else:
        u_emb = user_emb[user_inputs]
        return self.compute_score(u_emb, i_emb)

def compute_embeddings(self):
    enc_x = self.hgnn_encoder(
        self.user_embedding.weight,
        self.item_embedding.weight,
        self.group_embedding.weight,
        self.num_users,
        self.num_items,
    )

    user_emb, item_emb, group_emb = torch.split(
        enc_x, [self.num_users, self.num_items, self.num_groups]
    )

    return user_emb, item_emb, group_emb

def compute_score(self, u_emb, i_emb):
        element_emb = torch.mul(u_emb, i_emb)
        return self.predictor(torch.cat((element_emb, u_emb, i_emb), dim=1))

def get_member_list(self, group_inputs):

        member = [] 
        bsz = group_inputs.shape[0]
        for i in range(bsz):
            member.append(np.array(self.group_member_dict[group_inputs[i].item()]))
        return member
    
def get_member_mask(self, group_inputs, member_list):
        '''member_mask = [[mem of gp1 * mem of gp1]
                          [mem of gp2 * mem of gp2]
                          .........................]'''
        '''atten_mask = [[0, 0, 0, ...., 1, 1, 1]
                         [0, 0, 1, ...., 1, 1, 1]
                         ........................]'''

        bsz = group_inputs.shape[0]
        max_len = self.all_group_mem_prof.shape[1]
        atten_mask = np.ones((bsz, max_len))
        member_mask = [] # bsz * max_len
        for gp, mem in enumerate(member_list):
            cur_len = mem.shape[0]
            if (cur_len != max_len):
                member_mask.append(np.append(mem, np.zeros(max_len - cur_len))) 
            else:
                member_mask.append(mem)
            for l in range(cur_len):
                atten_mask[gp][l] = 0
        member_mask = torch.LongTensor(member_mask).to(self.device)

        return member_mask, atten_mask, max_len
    
def get_influence_embedding(self , member_masked, group_inputs, user_emb):

        mem_pro_mask= torch.Tensor(self.mem_pro_mask)[group_inputs].to(self.device)
        influence_mask = torch.Tensor(self.influence_mask)[group_inputs].to(self.device)
        group_mem_prof = self.all_group_mem_prof[group_inputs].to(self.device)
        influ_embedding = self.Influence_Prediction(user_emb, member_masked, group_mem_prof, mem_pro_mask, influence_mask)
        del group_mem_prof
        return influ_embedding
