class MLPAgent(th.nn.Module):
    def __init__(
        self,
        n_types: int,
        n_relations: int,
        n_actions: int,
        layers: int,
        embedding_dim: int,
        activation: th.nn.Module,
    ):
        super().__init__()
        # self.embedding = th.nn.Embedding(n_types, embedding_dim)
        # th.nn.init.constant_(self.embedding.weight, 1.0)
        self.mlp = th.nn.Sequential(
            th.nn.Linear(n_relations, embedding_dim),
            activation,
            # th.nn.Linear(embedding_dim, embedding_dim),
            # activation,
            # th.nn.Linear(embedding_dim, embedding_dim),
            # activation,
        )

        # self.embedders = th.nn.ModuleDict(
        #     {
        #         "button": th.nn.Linear(1, embedding_dim),
        #         "machine": th.nn.Linear(1, embedding_dim),
        #     }
        # )

        # th.nn.init.constant_(self.mlp[0].weight, 1.0)
        # th.nn.init.constant_(self.mlp[2].weight, 1.0)
        # th.nn.init.constant_(self.mlp[0].bias, 0.0)
        # th.nn.init.constant_(self.mlp[2].bias, 0.0)

        self.nullary_action = th.nn.Linear(embedding_dim, n_actions)
        self.unary_action = th.nn.Linear(embedding_dim, n_types)

        # th.nn.init.constant_(self.nullary_action.weight, 1.0)
        # th.nn.init.constant_(self.unary_action.weight, 1.0)
        # th.nn.init.constant_(self.nullary_action.bias, 0.0)
        # th.nn.init.constant_(self.unary_action.bias, 0.0)

        # self.unary_given_nullary_action = th.nn.Linear(
        #     embedding_dim, n_types * n_actions
        # )

    def forward(
        self, a: th.Tensor, x: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # e = self.embedding(x)
        # e = e.view(e.size(0), -1)

        # x = x.view(x.size(0), -1)

        logits = self.mlp(x)

        nullary_action = a[:, 0].unsqueeze(1)
        unary_action = a[:, 1].unsqueeze(1)

        p_nullary = th.nn.functional.softmax(self.nullary_action(logits), dim=-1)
        p_unary = th.nn.functional.softmax(self.unary_action(logits), dim=-1)

        # l_joint = self.unary_given_nullary_action(logits).view(
        #     -1,
        #     a.size(1),
        #     x.size(1),
        # )

        # conditional_logits = l_joint.gather(
        #     2, nullary_action.unsqueeze(-1).expand(-1, -1, x.size(1))
        # ).squeeze(1)

        # p_unary_given_nullary = th.nn.functional.softmax(
        #     self.unary_action(logits).view(-1, x.size(0), a.size(0)), dim=-1
        # )

        p_a_nullary = p_nullary.gather(1, nullary_action)
        p_a_unary = p_unary.gather(1, unary_action)

        # when nullary_action is 0, p_a_unary is 1
        p_a_unary = th.where(nullary_action == 0, th.ones_like(p_a_unary), p_a_unary)

        logprob = th.log(p_a_nullary * p_a_unary)

        return logprob

    def sample_action(self, x: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # x = x.view(x.size(0), -1)
        logits = self.mlp(x)

        p_nullary = th.nn.functional.softmax(self.nullary_action(logits), dim=-1)
        p_unary = th.nn.functional.softmax(self.unary_action(logits), dim=-1)

        threshold = 0.05
        # threshold and rescale
        p_unary = th.where(p_unary < threshold, th.zeros_like(p_unary), p_unary)
        p_unary = th.nn.functional.softmax(p_unary, dim=-1)

        nullary_action = (
            th.distributions.Categorical(p_nullary).sample()
            if not deterministic
            else th.argmax(p_nullary)
        )
        unary_action = (
            th.distributions.Categorical(p_unary).sample()
            if not deterministic
            else th.argmax(p_unary)
        )

        return th.stack([nullary_action, unary_action], dim=0)
