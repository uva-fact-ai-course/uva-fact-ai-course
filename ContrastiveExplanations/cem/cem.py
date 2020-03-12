""" This module implements the Contrastive Explanation Method in Pytorch.

Paper:  https://arxiv.org/abs/1802.07623
"""
import torch


class ContrastiveExplanationMethod:
    def __init__(
        self,
        classifier,
        autoencoder=None,
        kappa: float = 10.0,
        c_init: float = 10.0,
        c_converge: float = 0.1,
        beta: float = 0.1,
        gamma: float = 100.0,
        iterations: int = 1000,
        n_searches: int = 9,
        learning_rate: float = 0.01,
        verbose: bool = False,
        print_every: int = 100,
        input_shape: tuple = (1, 28, 28),
        device: str = "cpu",
    ):
        """
        Initialise the CEM model.

        classifier
            classification model to be explained.
        mode
            for pertinant negatives 'PN' or for pertinant positives 'PP'.
        autoencoder
            optional, autoencoder to be used for regularisation of the
            modifications to the explained samples.
        kappa
            confidence parameter used in the loss functions (eq. 2)
            and (eq. 4) in the original paper.
        const
            initial regularisation coefficient for the attack loss term.
        beta
            regularisation coefficent for the L1 term of the optimisation
            objective.
        gamma
            regularisation coefficient for the autoencoder term of the
            optimisation objective.
        iterations
            number of iterations in each search
        n_searches
            number of searches, also the number of times c gets adjusted
        learning_rate
            initial learning rate used to optimise the slack variable
        verbose
            print information during training
        print_every
            print frequency during training if verbose is true
        input_shape
            shape of single input sample, used to reshape for classifier
            and ae input
        device
            which device to run the CEM on
        """
        classifier.eval()
        classifier.to(device)
        if autoencoder:
            autoencoder.eval()
            autoencoder.to(device)
        self.classifier = classifier.forward_no_sm
        self.autoencoder = autoencoder
        self.kappa = kappa
        self.c_converge = c_converge
        self.c_init = c_init
        self.beta = beta
        self.gamma = gamma

        self.iterations = iterations
        self.n_searches = n_searches
        self.learning_rate = learning_rate

        self.verbose = verbose
        self.input_shape = input_shape
        self.device = device
        self.print_every = print_every

    def explain(self, orig, mode="PN"):
        """
        Determine pertinents for a given input sample.

        orig
            The original input sample to find the pertinent for.
        mode
            Either "PP" for pertinent positives or "PN" for pertinent
            negatives.

        """
        if mode not in ["PN", "PP"]:
            raise ValueError("Invalid mode. Please select either 'PP' or 'PN' as mode.")

        const = self.c_init
        step = 0

        orig = orig.view(*self.input_shape).to(self.device)

        best_loss = float("inf")
        best_delta = None

        orig_output = self.classifier(orig.view(-1, *self.input_shape))

        # mask for the originally selected label (t_0)
        target_mask = torch.zeros(orig_output.shape).to(self.device)
        target_mask[torch.arange(orig_output.shape[0]), torch.argmax(orig_output)] = 1

        # mask for the originally non-selected labels (i =/= t_0)
        nontarget_mask = torch.ones(orig_output.shape).to(self.device) - target_mask

        for search in range(self.n_searches):

            found_solution = False

            adv_img = torch.zeros(orig.shape).to(self.device)
            adv_img_slack = (
                torch.zeros(orig.shape).to(self.device).detach().requires_grad_(True)
            )

            # optimise for the slack variable y, with a square root decaying
            # learning rate
            optim = torch.optim.SGD([adv_img_slack], lr=self.learning_rate)

            for step in range(1, self.iterations + 1):

                # - Optimisation objective; (eq. 1) and (eq. 3) - #

                # reset the computational graph
                optim.zero_grad()
                adv_img_slack.requires_grad_(True)

                # Optimise for image + delta, this is more stable
                delta = orig - adv_img
                delta_slack = orig - adv_img_slack

                if mode == "PP":
                    perturbation_score = self.classifier(
                        delta_slack.view(-1, *self.input_shape)
                    )
                elif mode == "PN":
                    perturbation_score = self.classifier(
                        adv_img_slack.view(-1, *self.input_shape)
                    )

                target_lab_score = torch.max(target_mask * perturbation_score)
                nontarget_lab_score = torch.max(nontarget_mask * perturbation_score)

                # classification objective loss (eq. 2)
                if mode == "PP":
                    loss_attack = const * torch.max(
                        torch.tensor(0.0).to(self.device),
                        nontarget_lab_score - target_lab_score + self.kappa,
                    )
                elif mode == "PN":
                    loss_attack = const * torch.max(
                        torch.tensor(0.0).to(self.device),
                        -nontarget_lab_score + target_lab_score + self.kappa,
                    )

                # if the attack loss has converged to 0, a viable solution
                # has been found!
                if loss_attack.item() == 0:
                    found_solution = True

                # L2 regularisation term (eq. 1)
                l2_loss = torch.sum(delta ** 2)

                # reconstruction loss (eq. 1). reshape the image to fit ae
                # input, reshape the output of the autoencoder back. Since our
                # images are zero-mean, scale back to original MNIST range
                loss_ae = 0
                if mode == "PP" and callable(self.autoencoder):
                    ae_out = self.autoencoder(
                        delta_slack.view(-1, *self.input_shape) + 0.5
                    )
                    loss_ae = self.gamma * (
                        torch.norm(ae_out.view(*self.input_shape) - 0.5 - delta_slack)
                        ** 2
                    )
                elif mode == "PN" and callable(self.autoencoder):
                    ae_out = self.autoencoder(
                        adv_img_slack.view(-1, *self.input_shape) + 0.5
                    )
                    loss_ae = self.gamma * (
                        torch.norm(ae_out.view(*self.input_shape) - 0.5 - adv_img_slack)
                        ** 2
                    )

                # final optimisation objective
                loss_to_optimise = loss_attack + l2_loss + loss_ae

                # optimise for the slack variable, adjust lr
                loss_to_optimise.backward()
                optim.step()

                optim.param_groups[0]["lr"] = (self.learning_rate - 0.0) * (
                    1 - step / self.iterations
                ) ** 0.5

                adv_img_slack.requires_grad_(False)

                # - FISTA and corresponding update steps (eq. 5, 6) - #

                with torch.no_grad():

                    # Shrinkage thresholding function (eq. 7)
                    cond1 = torch.gt(adv_img_slack - orig, self.beta).type(torch.float)
                    cond2 = torch.le(torch.abs(adv_img_slack - orig), self.beta).type(
                        torch.float
                    )
                    cond3 = torch.lt(adv_img_slack - orig, -self.beta).type(torch.float)

                    # Ensure all delta values are between -0.5 and 0.5
                    upper = torch.min(
                        adv_img_slack - self.beta, torch.tensor(0.5).to(self.device)
                    )
                    lower = torch.max(
                        adv_img_slack + self.beta, torch.tensor(-0.5).to(self.device)
                    )

                    assign_adv_img = cond1 * upper + cond2 * orig + cond3 * lower

                    # Apply projection to the slack variable to obtain
                    # the value for delta (eq. 5)
                    cond4 = torch.gt(assign_adv_img - orig, 0).type(torch.float)
                    cond5 = torch.le(assign_adv_img - orig, 0).type(torch.float)
                    if mode == "PP":
                        assign_adv_img = cond5 * assign_adv_img + cond4 * orig
                    elif mode == "PN":
                        assign_adv_img = cond4 * assign_adv_img + cond5 * orig

                    # Apply momentum from previous delta and projection step
                    # to obtain the value for the slack variable (eq. 6)
                    mom = (step / (step + 3)) * (assign_adv_img - adv_img)
                    assign_adv_img_slack = assign_adv_img + mom
                    cond6 = torch.gt(assign_adv_img_slack - orig, 0).type(torch.float)
                    cond7 = torch.le(assign_adv_img_slack - orig, 0).type(torch.float)

                    # For PP only retain delta values that are smaller than
                    # the corresponding feature values in the original image
                    if mode == "PP":
                        assign_adv_img_slack = (
                            cond7 * assign_adv_img_slack + cond6 * orig
                        )
                    # For PN only retain delta values that are larger than
                    # the corresponding feature values in the original image
                    elif mode == "PN":
                        assign_adv_img_slack = (
                            cond6 * assign_adv_img_slack + cond7 * orig
                        )

                    adv_img.data.copy_(assign_adv_img)
                    adv_img_slack.data.copy_(assign_adv_img_slack)

                    # check if the found delta solves the classification
                    # problem, retain if it is the most regularised solution
                    if loss_attack.item() == 0:
                        if loss_to_optimise < best_loss:

                            best_loss = loss_to_optimise
                            best_delta = adv_img.detach().clone()

                            if self.verbose:
                                print(
                                    "new best delta found, loss: {}".format(
                                        loss_to_optimise
                                    )
                                )

                    if self.verbose and not (step % self.print_every):
                        print_vars = (
                            search,
                            step,
                            const,
                            loss_to_optimise,
                            found_solution,
                        )
                        print(
                            (
                                "search: {} iteration: {} c: {:.2f}"
                                + " loss: {:.2f} solution: {}"
                            ).format(*print_vars)
                        )

            # If in this search a solution has been found we can decrease the
            # weight of the attack loss to increase the regularisation, else
            # increase c to decrease regularisation
            if found_solution:
                const = (self.c_converge + const) / 2
            else:
                const *= 10

        return best_delta
