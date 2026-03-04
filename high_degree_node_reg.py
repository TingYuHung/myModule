def high_degree_node_reg(self):
        init_x = torch.concat(
            (
                self.user_embedding.weight,
                self.item_embedding.weight,
                self.group_embedding.weight,
            ),
            dim=0,
        )
        # mask
        (user_emb, item_emb, group_emb), mask_idxs = self.mask_encoding(init_x)
        # encoder
        enc_x = self.hgnn_encoder(
            user_emb, item_emb, group_emb, self.num_users, self.num_items
        )
        # regenerate a new X to be masked
        masked_x = self.regenerate(enc_x)
        masked_x[mask_idxs] = 0.0 + 1e-8
        # decoder
        dec_x = self.mlp_decoder(masked_x)
        return (
            self.scaled_cosine_error(
                init_x[mask_idxs], dec_x[mask_idxs], self.sce_alpha
            ),
            enc_x
          
