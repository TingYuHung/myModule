class HGNN_Encoder(nn.Module):
    def __init__(
        self,
        user_hyper_graph,
        item_hyper_graph,
        full_hyper,
        emb_dim,
        num_layers, 
        device,
    ):
        super(HGNN_Encoder, self).__init__()
        self.user_hyper, self.item_hyper, self.full_hyper_graph = (
            user_hyper_graph,
            item_hyper_graph,
            full_hyper,
        )
        self.hgnns = [HyperConvLayer(emb_dim).to(device) for _ in range(num_layers)]

    def forward(self, user_emb, item_emb, group_emb, num_users, num_items):
        init_ui_emb = torch.cat([user_emb, item_emb], dim=0)
        init_g_emb = group_emb
        final = [init_ui_emb]
        final_he = [init_g_emb]
        for hgnn in self.hgnns:
            ui_emb, g_emb = hgnn(
                user_emb,
                item_emb,
                init_g_emb,
                self.user_hyper,
                self.item_hyper,
                self.full_hyper_graph,
            )
            final.append(ui_emb)
            final_he.append(g_emb)

            user_emb, item_emb = torch.split(ui_emb, [num_users, num_items])

        final_emb = torch.sum(torch.stack(final), dim=0)
        final_he = torch.sum(torch.stack(final_he), dim=0)
        return torch.concat((final_emb, final_he), dim=0)
