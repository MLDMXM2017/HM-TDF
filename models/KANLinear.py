# Credits to: https://github.com/Blealtan/efficient-kan and https://github.com/KindXiaoming/pykan
# @inproceedings{HM-TDF,  
#   title={Hard Sample Mining-Based Tongue Diagnosis Framework for Fatty Liver Disease Severity Classification Using Kolmogorov-Arnold Network},  
#   link={https://github.com/MLDMXM2017/HM-TDF}  
# }  
import torch
import torch.nn.functional as F
import math

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

        self.regularization_loss_activation_ratio = 1 / math.sqrt(self.in_features) if self.out_features > 1 else 0
        self.regularization_loss_entropy_ratio = 1 / math.log(self.out_features) if self.out_features > 1 else 0

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum() 
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())  
        return (
            regularize_activation * regularization_loss_activation * self.regularization_loss_activation_ratio
            + regularize_entropy * regularization_loss_entropy * self.regularization_loss_entropy_ratio
        )

    def plot_forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_edge = self.base_activation(x).unsqueeze(1) * self.base_weight.unsqueeze(0) # [batch, out, in]
        base_output = base_edge.sum(-1) # [batch, out]
        # base_output = F.linear(self.base_activation(x), self.base_weight) # [batch, out]

        splines_edge = (self.b_splines(x).unsqueeze(1) * self.scaled_spline_weight.unsqueeze(0)).sum(-1) # [batch, out, in]
        spline_output = splines_edge.sum(-1)

        out_edge = base_edge + splines_edge
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)

        input_range = torch.std(x, dim=0, keepdim=True) + 0.1
        output_range = torch.std(out_edge, dim=0)
        acts_scale = output_range / input_range

        return output, base_edge, splines_edge, out_edge, acts_scale


    def plot(self, acts, folder="./figures", beta=3, 
             top_k_input=-1, top_out_index=None, top_width=-1, mask=False, mode="supervised", 
             fig_size=10, scale=0.5, tick=False, sample=False, 
             in_vars=None, out_vars=None, title=None, split=[]):
        
        """
            plot key pathways
        """
        
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        if not os.path.exists(folder):
            os.makedirs(folder)

        output, base_edge, splines_edge, out_edge, acts_scale = self.plot_forward(acts)
        acts = [acts]
        spline_postacts = [out_edge]
        acts_scale = [acts_scale]

        def score2alpha(score):
            return np.tanh(beta * score)
        
        if top_k_input < 0 or top_k_input > self.in_features:
            top_k_input = self.in_features
        if not isinstance(top_out_index, np.ndarray):
            top_out_index = np.arange(self.out_features)
        
        # top_k_index = [np.unique(np.argsort(-score.cpu().detach().numpy())[:, :top_k_input].reshape(-1)) for score in acts_scale] + [np.arange(self.out_features)]
        from collections import Counter
        if len(split):
            split_i = 0
            split_top_k_input = int(top_k_input/len(split))
            top_in_index_hub = []
            for i in range(len(split)):
                counter = Counter(np.argsort(-acts_scale[0][:, split_i:split_i+split[i]].cpu().detach().numpy())[:, :split_top_k_input].reshape(-1))
                top_in_index_list = [element for element, _ in counter.most_common(split_top_k_input)]
                top_in_index_hub.append(np.sort(np.asarray(top_in_index_list)) + split_i)
                split_i += split[i]
            top_in_index = [np.concatenate(top_in_index_hub), top_out_index]
        else:
            counter = Counter(np.argsort(-acts_scale[0].cpu().detach().numpy())[:, :top_k_input].reshape(-1))
            top_in_index_list = [element for element, _ in counter.most_common(top_k_input)]
            top_in_index = [np.sort(np.asarray(top_in_index_list)), top_out_index]
        
        alpha = [score2alpha(score.cpu().detach().numpy()) for score in acts_scale]

        depth = 1
        for l in range(depth):
            w_large = 2.0
            for i in top_in_index[l]:
                for j in top_in_index[l+1]:
                    
                    rank = torch.argsort(acts[l][:, i])
                    fig, ax = plt.subplots(figsize=(w_large, w_large))

                    num = rank.shape[0]

                    color = "black"
                    alpha_mask = 1

                    if tick == True:
                        ax.tick_params(axis="y", direction="in", pad=-22, labelsize=50)
                        ax.tick_params(axis="x", direction="in", pad=-15, labelsize=50)
                        x_min, x_max, y_min, y_max = self.get_range(l, i, j, verbose=False)
                        plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                        plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    if alpha_mask == 1:
                        plt.gca().patch.set_edgecolor('black')
                    else:
                        plt.gca().patch.set_edgecolor('white')
                    plt.gca().patch.set_linewidth(1.5)
                    # plt.axis('off')

                    plt.plot(acts[l][:, i][rank].cpu().detach().numpy(), spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5)
                    if sample == True:
                        plt.scatter(self.acts[l][:, i][rank].cpu().detach().numpy(), spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, s=400 * scale ** 2)
                    plt.gca().spines[:].set_color(color)

                    plt.savefig(f'{folder}/sp_{l}_{i}_{j}.png', bbox_inches="tight", dpi=200) # dpi=400
                    plt.close()

        width = np.array([top_index.shape[0] for top_index in top_in_index]) # np.array(self.width)
        A = 1
        y0 = 0.2  # 0.4

        # plt.figure(figsize=(5,5*(neuron_depth-1)*y0))
        neuron_depth = len(width)
        min_spacing = A / np.maximum(np.max(width), 5)

        max_neuron = np.max(width)
        max_num_weights = np.max(width[:-1] * width[1:])
        y1 = 0.6 / np.maximum(max_num_weights, 3)

        fig, ax = plt.subplots(figsize=(fig_size * scale, fig_size * scale * (neuron_depth - 1) * y0))
        # fig, ax = plt.subplots(figsize=(5,5*(neuron_depth-1)*y0))

        # plot scatters and lines
        for l in range(neuron_depth):
            top_index_i = top_in_index[l]
            n = width[l]
            spacing = A / n
            for i in range(n):
                # plot neurons
                if len(split) and l:
                    plt.scatter(1 / (2 * n) + i / n       / 2     , l * y0, s=min_spacing ** 2 * 10000 * scale ** 2, color='black')
                else:
                    plt.scatter(1 / (2 * n) + i / n, l * y0, s=min_spacing ** 2 * 10000 * scale ** 2, color='black')
                if l < neuron_depth - 1:
                    # plot connections
                    top_index_j = top_in_index[l+1]
                    n_next = width[l + 1]
                    N = n * n_next
                    for j in range(n_next):
                        id_ = i * n_next + j

                        color = "black"
                        alpha_mask = 1.

                        if mask == True:
                            plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * y0, (l + 1 / 2) * y0 - y1], color=color, lw=2 * scale, alpha=alpha[l][top_index_j[j]][top_index_i[i]])
                            plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [(l + 1 / 2) * y0 + y1, (l + 1) * y0], color=color, lw=2 * scale, alpha=alpha[l][top_index_j[j]][top_index_i[i]])
                        else:
                            if len(split):
                                plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * y0, (l + 1 / 2) * y0 - y1], color=color, lw=2 * scale, alpha=alpha[l][top_index_j[j]][top_index_i[i]] * alpha_mask)
                                plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next     / 2    ], [(l + 1 / 2) * y0 + y1, (l + 1) * y0], color=color, lw=2 * scale, alpha=alpha[l][top_index_j[j]][top_index_i[i]] * alpha_mask)
                            else:
                                plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * y0, (l + 1 / 2) * y0 - y1], color=color, lw=2 * scale, alpha=alpha[l][top_index_j[j]][top_index_i[i]] * alpha_mask)
                                plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [(l + 1 / 2) * y0 + y1, (l + 1) * y0], color=color, lw=2 * scale, alpha=alpha[l][top_index_j[j]][top_index_i[i]] * alpha_mask)

            plt.xlim(0, 1)
            plt.ylim(-0.1 * y0, (neuron_depth - 1 + 0.1) * y0)
        # -- Transformation functions
        DC_to_FC = ax.transData.transform
        FC_to_NFC = fig.transFigure.inverted().transform
        # -- Take data coordinates and transform them to normalized figure coordinates
        DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

        plt.axis('off')

        # plot splines
        for l in range(neuron_depth - 1):
            n = width[l]
            in_index = top_in_index[l]
            out_index = top_in_index[l+1]
            for i in range(n):
                n_next = width[l + 1]
                N = n * n_next
                for j in range(n_next):
                    id_ = i * n_next + j
                    im = plt.imread(f'{folder}/sp_{l}_{in_index[i]}_{out_index[j]}.png')
                    left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                    right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                    bottom = DC_to_NFC([0, (l + 1 / 2) * y0 - y1])[1]
                    up = DC_to_NFC([0, (l + 1 / 2) * y0 + y1])[1]
                    newax = fig.add_axes([left, bottom, right - left, up - bottom])
                    # newax = fig.add_axes([1/(2*N)+id_/N-y1, (l+1/2)*y0-y1, y1, y1], anchor='NE')
                    if mask == False:
                        newax.imshow(im, alpha=alpha[l][out_index[j]][in_index[i]])
                    else:
                        ### make sure to run model.prune() first to compute mask ###
                        newax.imshow(im, alpha=alpha[l][out_index[j]][in_index[i]])
                    newax.axis('off')


        if in_vars != None:
            n = width[0]
            for i in range(n):
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, in_vars[i], fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')
        else:
            n = width[0]
            for i in range(n):
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, f'{top_in_index[0][i]}', fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')

        if out_vars != None:
            n = width[-1]
            for i in range(n):
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), y0 * (len(width) - 1) + 0.1, out_vars[i], fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')
        else:
            n = width[-1]
            for i in range(n):
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), y0 * (len(width) - 1) + 0.1, f'{top_in_index[-1][i]}', fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')


        if title != None:
            plt.gcf().get_axes()[0].text(0.5, y0 * (len(width) - 1) + 0.2, title, fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')

        import time
        plt.savefig(f'{folder}/{time.time()}.png', bbox_inches="tight", dpi=600)

        return top_in_index
