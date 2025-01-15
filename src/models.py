import torch
import torch.nn as nn
import torch.nn.functional as F


class NetParent(nn.Module):
    """
    Mostly a QOL superclass
    Creates a parent class that has reset_parameters implemented and .device
    so I don't have to re-write it to each child class and can just inherit it
    """

    def __init__(self):
        super(NetParent, self).__init__()
        # device is cpu by default
        self.device = 'cpu'
        self.counter = 0

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight.data)

    @staticmethod
    def reset_weight(layer):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    def reset_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for child in self.children():
            if hasattr(child, 'children'):
                for sublayer in child.children():
                    self.reset_weight(sublayer)
            if hasattr(child, 'reset_parameters'):
                self.reset_weight(child)

    def to(self, device):
        # Work around, so we can get model.device for all NetParent
        #
        super(NetParent, self).to(device)
        self.device = device

    def increment_counter(self):
        self.counter += 1
        for c in self.children():
            if hasattr(c, 'counter') and hasattr(c, 'increment_counter'):
                c.increment_counter()


class StandardizerSequence(nn.Module):
    def __init__(self, n_feats=20):
        super(StandardizerSequence, self).__init__()
        # Here using 20 because 20 AA alphabet. With this implementation, it shouldn't need custom state_dict fct
        self.mu = nn.Parameter(torch.zeros(n_feats), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(n_feats), requires_grad=False)
        self.fitted = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.n_feats = n_feats
        self.dimensions = None

    def fit(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        assert self.training, 'Can not fit while in eval mode. Please set model to training mode'
        with torch.no_grad():
            masked_values = x_tensor * x_mask
            mu = (torch.sum(masked_values, dim=1) / torch.sum(x_mask, dim=1))
            sigma = (torch.sqrt(torch.sum((masked_values - mu.unsqueeze(1)) ** 2, dim=1) / torch.sum(x_mask, dim=1)))
            self.mu.data.copy_(mu.mean(dim=0))
            sigma = sigma.mean(dim=0)
            sigma[torch.where(sigma == 0)] = 1e-12
            self.sigma.data.copy_(sigma)
            self.fitted.data = torch.tensor(True)

    def forward(self, x):
        assert self.fitted.data, 'StandardizerSequence has not been fitted. Please fit to x_train'
        with torch.no_grad():
            # Flatten to 2d if needed
            x = (self.view_3d_to_2d(x) - self.mu) / self.sigma
            # Return to 3d if needed
            return self.view_2d_to_3d(x)

    def recover(self, x):
        # assert self.fitted, 'StandardizerSequence has not been fitted. Please fit to x_train'
        with torch.no_grad():
            # Flatten to 2d if needed
            x = self.view_3d_to_2d(x)
            # Return to original scale by multiplying with sigma and adding mu
            x = x * self.sigma + self.mu
            # Return to 3d if needed
            return self.view_2d_to_3d(x)

    def reset_parameters(self, **kwargs):
        with torch.no_grad():
            self.mu.data.copy_(torch.zeros(self.n_feats, device=self.device))
            self.sigma.data.copy_(torch.ones(self.n_feats, device=self.device))
            self.fitted.data = torch.tensor(False)

    def view_3d_to_2d(self, x):
        with torch.no_grad():
            if len(x.shape) == 3:
                self.dimensions = (x.shape[0], x.shape[1], x.shape[2])
                return x.view(-1, x.shape[2])
            else:
                return x

    def view_2d_to_3d(self, x):
        with torch.no_grad():
            if len(x.shape) == 2 and self.dimensions is not None:
                return x.view(self.dimensions[0], self.dimensions[1], self.dimensions[2])
            else:
                return x

    def to(self, device):
        super(StandardizerSequence, self).to(device)
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)


class StandardizerFeatures(nn.Module):
    def __init__(self, n_feats=2):
        super(StandardizerFeatures, self).__init__()
        self.mu = nn.Parameter(torch.zeros(n_feats), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(n_feats), requires_grad=False)
        self.fitted = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.n_feats = n_feats

    def fit(self, x_features: torch.Tensor):
        """ Will consider the mask (padded position) and ignore them before computing the mean/std
        Args:
            x_features:

        Returns:
            None
        """
        assert self.training, 'Can not fit while in eval mode. Please set model to training mode'
        with torch.no_grad():
            self.mu.data.copy_(x_features.mean(dim=(0, 1)))
            self.sigma.data.copy_(x_features.std(dim=(0, 1)))
            # Fix issues with sigma=0 that would cause a division by 0 and return NaNs
            self.sigma.data[torch.where(self.sigma.data == 0)] = 1e-12
            self.fitted.data = torch.tensor(True)

    def forward(self, x):
        assert self.fitted.data, 'StandardizerSequence has not been fitted. Please fit to x_train'
        with torch.no_grad():
            return x - self.mu / self.sigma

    def reset_parameters(self, **kwargs):
        with torch.no_grad():
            self.mu.data.copy(torch.zeros(self.n_feats))
            self.sigma.data.copy(torch.ones(self.n_feats))
            self.fitted.data = torch.tensor(False)

    def to(self, device):
        super(StandardizerFeatures, self).to(device)
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)


class StandardizerSequenceVector(nn.Module):
    def __init__(self, input_dim=20, max_len=12):
        super(StandardizerSequenceVector, self).__init__()
        self.mu = nn.Parameter(torch.zeros((max_len, input_dim)), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones((max_len, input_dim)), requires_grad=False)
        self.fitted = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.input_dim = input_dim
        self.max_len = max_len

    def fit(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        assert self.training, 'Can not fit while in eval mode. Please set model to training mode'
        with torch.no_grad():
            masked_values = x_tensor * x_mask
            mu = masked_values.mean(dim=0)
            sigma = masked_values.std(dim=0)
            sigma[torch.where(sigma == 0)] = 1e-12
            self.mu.data.copy_(mu)
            self.sigma.data.copy_(sigma)
            self.fitted.data = torch.tensor(True)

    def forward(self, x):
        assert self.fitted, 'Standardizer not fitted!'
        return (x - self.mu) / self.sigma

    def reset_parameters(self, **kwargs):
        with torch.no_grad():
            self.mu.data.copy_(torch.zeros((self.max_len, self.input_dim)))
            self.sigma.data.copy_(torch.ones((self.max_len, self.input_dim)))
            self.fitted.data = torch.tensor(False)

    def to(self, device):
        super(StandardizerSequenceVector, self).to(device)
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)


class StdBypass(nn.Module):
    def __init__(self, **kwargs):
        super(StdBypass, self).__init__()
        self.requires_grad = False
        self.bypass = nn.Identity(**kwargs)
        self.fitted = False
        self.mu = None
        self.sigma = None

    def forward(self, x_tensor):
        """
        Args:
            x:
            x_mask: x_mask here exists for compatibility purposes


        Returns:

        """

        return x_tensor

    def fit(self, x_tensor, x_mask):
        """
        Args:
            x:
            x_mask: x_mask here exists for compatibility purposes


        Returns:

        """
        self.fitted = True
        return x_tensor


class NNAlignSinglePass(NetParent):
    """
    NNAlign implementation with a single forward pass where best score selection + indexing is done in one pass.
    """

    def __init__(self, n_hidden, window_size,
                 activation, batchnorm=False,
                 dropout=0.0):
        super(NNAlignSinglePass, self).__init__()
        self.matrix_dim = 20
        self.window_size = window_size
        self.n_hidden = n_hidden
        self.in_layer = nn.Linear(self.window_size * self.matrix_dim, n_hidden)
        self.out_layer = nn.Linear(n_hidden, 1)
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(n_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.act = activation

    def forward(self, x_tensor: torch.Tensor, x_mask: torch.tensor):

        """
        Single forward pass for layers + best score selection without w/o grad
        Args:
            x_mask:
            x_tensor:

        Returns:

        """
        # FIRST FORWARD PASS: best scoring selection, with no grad
        z = self.in_layer(x_tensor)  # Inlayer
        # Flip dimensions to allow for batchnorm then flip back
        if self.batchnorm:
            z = self.bn1(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden)) \
                .view(-1, x_tensor.shape[1], self.n_hidden)
        z = self.dropout(z)
        z = self.act(z)
        z = self.out_layer(z)  # Out Layer for prediction

        # NNAlign selecting the max score here
        with torch.no_grad():
            # Here, use sigmoid to set values to 0,1 before masking
            # only for index selection, Z will be returned as logits
            max_idx = torch.mul(F.sigmoid(z), x_mask).argmax(dim=1).unsqueeze(1)

        z = torch.gather(z, 1, max_idx).squeeze(1)  # Indexing the best submers
        return z

    def predict(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        """Works like forward but also returns the index (for the motif selection/return)

        This should be done with torch no_grad as this shouldn't be used during/for training
        Also here does the sigmoid to return scores within [0, 1] on Z
        Args:
            x_tensor: torch.Tensor, the input tensor (i.e. encoded sequences)
            x_mask: torch.Tensor, to mask padded positions
        Returns:
            z: torch.Tensor, the best scoring K-mer for each of the input in X
            max_idx: torch.Tensor, the best indices corresponding to the best K-mer,
                     used to find the predicted core
        """
        with torch.no_grad():
            z = self.in_layer(x_tensor)
            if self.batchnorm:
                z = self.bn1(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden)) \
                    .view(-1, x_tensor.shape[1], self.n_hidden)
            z = self.act(self.dropout(z))
            z = self.out_layer(z)
            # Do the same trick where the padded positions are removed prior to selecting index
            max_idx = torch.mul(F.sigmoid(z), x_mask).argmax(dim=1).unsqueeze(1)
            # Additionally run sigmoid on z so that it returns proba in range [0, 1]
            z = F.sigmoid(torch.gather(z, 1, max_idx).squeeze(1))
            return z, max_idx

    def predict_logits(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        """ QOL method to return the predictions without Sigmoid + return the indices
        To be used elsewhere down the line (in EF model)

        Args:
            x_tensor:
            x_mask:

        Returns:

        """
        with torch.no_grad():
            z = self.in_layer(x_tensor)
            if self.batchnorm:
                z = self.bn1(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden)) \
                    .view(-1, x_tensor.shape[1], self.n_hidden)
            z = self.act(self.dropout(z))
            z = self.out_layer(z)
            # Do the same trick where the padded positions are removed prior to selecting index
            max_idx = torch.mul(F.sigmoid(z), x_mask).argmax(dim=1).unsqueeze(1)
            # Additionally run sigmoid on z so that it returns proba in range [0, 1]
            z = torch.gather(z, 1, max_idx).squeeze(1)
            return z, max_idx


class NNAlignEFSinglePass(NetParent):
    """
    NNAlign implementation with a single forward pass where best score selection + indexing is done in one pass.

    """

    def __init__(self, n_hidden, n_hidden_2, window_size,
                 activation=nn.ReLU(), feat_dim=0, pseudoseq_dim=0, batchnorm=False,
                 dropout=0.0, standardize=False,
                 add_hidden_layer=False, add_structure=False):
        super(NNAlignEFSinglePass, self).__init__()
        if add_structure:
            self.matrix_dim = 25
        else:
            self.matrix_dim = 20
        self.window_size = window_size
        self.n_hidden = n_hidden
        self.n_hidden_2 = n_hidden_2
        self.feat_dim = feat_dim
        self.pseudoseq_dim = pseudoseq_dim
        self.add_hidden_layer = add_hidden_layer
        # Input layer
        self.in_layer = nn.Linear(self.window_size * self.matrix_dim + feat_dim + pseudoseq_dim, n_hidden)
        # Additional hidden layer if use_second_hidden_layer is True
        if add_hidden_layer:
            self.hidden_layer = nn.Linear(n_hidden, n_hidden_2)
            if batchnorm:
                self.bnh = nn.BatchNorm1d(n_hidden_2)
            self.out_layer = nn.Linear(n_hidden_2, 1)
        else:
            self.out_layer = nn.Linear(n_hidden, 1)
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(n_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.act = activation
        self.standardizer_sequence = StandardizerSequence(
            n_feats=(self.matrix_dim * window_size) + feat_dim) if standardize else StdBypass()
        # For mhc pseudosequences, extrafeat_dim would be 680 (34x20, flattened)
        self.standardizer_features = StandardizerFeatures(n_feats=self.pseudoseq_dim) if standardize else StdBypass()

    def fit_standardizer(self, x_tensor, x_mask, x_feats=None):
        assert self.training, 'Must be in training mode to fit!'
        with torch.no_grad():
            self.standardizer_sequence.fit(x_tensor, x_mask)
            if x_feats is not None:
                self.standardizer_features.fit(self.reshape_features(x_tensor, x_feats))

    @staticmethod
    def reshape_features(x_tensor, x_feats):
        """
        ON THE FLY PART
        Reshapes and repeats the feature tensors in order to concatenate them to the x_tensor
        This is to "duplicate" MHC pseudosequences on the fly ; Reshapes x_feats into x_tensor's shape
        """
        return x_feats.unsqueeze(1).repeat(1, x_tensor.shape[1], 1)

    def forward(self, x_tensor: torch.Tensor, x_mask: torch.tensor, x_feats: torch.tensor = None):

        """
        Single forward pass for layers + best score selection without w/o grad
        Here the final activation is a sigmoid because we are training on BA data with values in [0,1] and using MSE
        Args:
            x_mask:
            x_tensor:

        Returns:

        """
        z, _ = self.nnalign_logits(x_tensor, x_mask, x_feats)
        return F.sigmoid(z)

    def predict(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_feats=None):
        """Works like forward but also returns the index (for the motif selection/return)

        This should be done with torch no_grad as this shouldn't be used during/for training
        Also here does the sigmoid to return scores within [0, 1] on Z
        Args:
            x_tensor: torch.Tensor, the input tensor (i.e. encoded sequences)
            x_mask: torch.Tensor, to mask padded positions
        Returns:
            z: torch.Tensor, the best scoring K-mer for each of the input in X
            max_idx: torch.Tensor, the best indices corresponding to the best K-mer,
                     used to find the predicted core
        """
        with torch.no_grad():
            z, max_idx = self.nnalign_logits(x_tensor, x_mask, x_feats)
            return F.sigmoid(z), max_idx

    def nnalign_logits(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_feats=None):
        """ QOL method to return the predictions without Sigmoid + return the indices
        To be used elsewhere down the line (in EF model)

        Args:
            x_tensor:
            x_mask:

        Returns:

        """
        # No grad for the standardizer operations
        with torch.no_grad():
            x_tensor = self.standardizer_sequence(x_tensor)
            if x_feats is not None:
                # Takes the flattened x_features tensor and repeats it for each icore
                x_feats = self.standardizer_features(self.reshape_features(x_tensor, x_feats))
                x_tensor = torch.cat([x_tensor, x_feats], dim=2)

        # Runs input layer and batchnorm if true, order: Layer -> BN -> DO -> ReLU
        z = self.in_layer(x_tensor)
        if self.batchnorm:
            z = self.bn1(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden)) \
                .view(-1, x_tensor.shape[1], self.n_hidden)
        z = self.act(self.dropout(z))
        # Additional hidden layer
        if self.add_hidden_layer:
            z = self.hidden_layer(z)
            if self.batchnorm:
                z = self.bnh(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden_2)) \
                    .view(-1, x_tensor.shape[1], self.n_hidden_2)
            z = self.act(self.dropout(z))
            z = self.out_layer(z)
        else:
            z = self.out_layer(z)  # Out Layer for prediction

        # NNAlign selecting the max score here
        with torch.no_grad():
            # Here, use sigmoid to set values to 0,1 before masking
            # only for index selection, Z will be returned as logits
            # Do the same trick where the padded positions are removed prior to selecting index
            max_idx = torch.mul(F.sigmoid(z), x_mask).argmax(dim=1).unsqueeze(1)
        # Additionally run sigmoid on z so that it returns proba in range [0, 1]
        z = torch.gather(z, 1, max_idx).squeeze(1)
        return z, max_idx

    def __old_forward(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_feats=None):
        # FIRST FORWARD PASS: best scoring selection, with no grad

        # Here concatenate whatever extra features (like the flattened MHC pseudosequence)
        with torch.no_grad():
            x_tensor = self.standardizer_sequence(x_tensor)
            if x_feats is not None:
                # Takes the flattened x_features tensor and repeats it for each icore
                x_feats = self.standardizer_features(self.reshape_features(x_tensor, x_feats))
                x_tensor = torch.cat([x_tensor, x_feats], dim=2)

        z = self.in_layer(x_tensor)
        # Flip dimensions to allow for batchnorm then flip back
        if self.batchnorm:
            z = self.bn1(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden)) \
                .view(-1, x_tensor.shape[1], self.n_hidden)
        z = self.dropout(z)
        z = self.act(z)
        # Additional hidden layer
        if self.add_hidden_layer:
            z = self.hidden_layer(z)
            z = self.act(z)
            z = self.out_layer(z)
        else:
            z = self.out_layer(z)  # Out Layer for prediction

        # NNAlign selecting the max score here
        with torch.no_grad():
            # Here, use sigmoid to set values to 0,1 before masking
            # only for index selection, Z will be returned as logits
            max_idx = torch.mul(F.sigmoid(z), x_mask).argmax(dim=1).unsqueeze(1)

        z = F.sigmoid(torch.gather(z, 1, max_idx).squeeze(1))  # Indexing the best submers
        return z


class NNAlignEFTwoStage(NetParent):
    """
    NNAlign implementation with a single forward pass where best score selection + indexing is done in one pass.
    TODO : Here this model has a "two-stage" behaviour with a first stage doing the core selection (up to z = torch.gather(argmax(z)...))
           Then, the second stage is concatenating to the structural features and running one additional layer for the output
    """

    def __init__(self, n_hidden, n_hidden_2, window_size,
                 activation=nn.ReLU(), feat_dim=0, pseudoseq_dim=0, batchnorm=False,
                 dropout=0.0, standardize=False,
                 add_hidden_layer=False, add_structure=False, add_mean_structure=False):
        super(NNAlignEFTwoStage, self).__init__()
        if add_structure:
            self.matrix_dim = 25
        else:
            self.matrix_dim = 20
        self.window_size = window_size
        self.n_hidden = n_hidden
        self.n_hidden_2 = n_hidden_2
        self.feat_dim = feat_dim
        self.pseudoseq_dim = pseudoseq_dim
        self.add_hidden_layer = add_hidden_layer
        self.add_mean_structure = add_mean_structure
        # Input layer
        self.in_layer = nn.Linear(self.window_size * self.matrix_dim + feat_dim + pseudoseq_dim, n_hidden)
        # Additional hidden layer if use_second_hidden_layer is True
        if add_hidden_layer:
            self.hidden_layer = nn.Linear(n_hidden, n_hidden_2)
            if batchnorm:
                self.bnh = nn.BatchNorm1d(n_hidden_2)
            self.out_layer = nn.Linear(n_hidden_2, 1)
        else:
            self.out_layer = nn.Linear(n_hidden, 1)
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(n_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.act = activation
        self.standardizer_sequence = StandardizerSequence(
            n_feats=(self.matrix_dim * window_size) + feat_dim) if standardize else StdBypass()
        # For mhc pseudosequences, extrafeat_dim would be 680 (34x20, flattened)
        self.standardizer_features = StandardizerFeatures(n_feats=self.pseudoseq_dim) if standardize else StdBypass()
        self.standardizer_structures = StandardizerFeatures(n_feats=5) if (
                standardize and add_mean_structure) else StdBypass()
        self.final_layer = nn.Linear(6, 1)  # 5 mean struct features + 1 core max score = 6

    def fit_standardizer(self, x_tensor, x_mask, x_feats=None):
        assert self.training, 'Must be in training mode to fit!'
        with torch.no_grad():
            if self.add_mean_structure:
                x_struct = x_tensor[:, 0, -5:].squeeze(1)
                self.standardizer_structures.fit(x_struct)
                x_tensor = x_tensor[:, ::-5]

            self.standardizer_sequence.fit(x_tensor, x_mask)
            if x_feats is not None:
                self.standardizer_features.fit(self.reshape_features(x_tensor, x_feats))

    @staticmethod
    def reshape_features(x_tensor, x_feats):
        """
        ON THE FLY PART
        Reshapes and repeats the feature tensors in order to concatenate them to the x_tensor
        This is to "duplicate" MHC pseudosequences on the fly ; Reshapes x_feats into x_tensor's shape
        """
        return x_feats.unsqueeze(1).repeat(1, x_tensor.shape[1], 1)

    def forward(self, x_tensor: torch.Tensor, x_mask: torch.tensor, x_feats: torch.tensor = None):

        """
        Single forward pass for layers + best score selection without w/o grad
        Here the final activation is a sigmoid because we are training on BA data with values in [0,1] and using MSE
        Args:
            x_mask:
            x_tensor:

        Returns:

        """
        # Doesn't return the max_idx in forward
        z, _ = self.nnalign_logits(x_tensor, x_mask, x_feats)
        return F.sigmoid(z)

    def predict(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_feats=None):
        """Works like forward but also returns the index (for the motif selection/return)

        This should be done with torch no_grad as this shouldn't be used during/for training
        Also here does the sigmoid to return scores within [0, 1] on Z
        Args:
            x_tensor: torch.Tensor, the input tensor (i.e. encoded sequences)
            x_mask: torch.Tensor, to mask padded positions
        Returns:
            z: torch.Tensor, the best scoring K-mer for each of the input in X
            max_idx: torch.Tensor, the best indices corresponding to the best K-mer,
                     used to find the predicted core
        """
        with torch.no_grad():
            z, max_idx = self.nnalign_logits(x_tensor, x_mask, x_feats)
            return F.sigmoid(z), max_idx

    def nnalign_logits(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_feats=None):
        """ QOL method to return the predictions without Sigmoid + return the indices
        To be used elsewhere down the line (in EF model)

        Args:
            x_tensor:
            x_mask:

        Returns:

        """
        # No gradient for the standardizer operations
        # Here concatenate whatever extra features (like the flattened MHC pseudosequence)
        with torch.no_grad():
            # TODO : Here, it's a two-stage model ; Unsafe but quick way to add the features is to split from x_feat instead of re-writing code
            #        Assumes that the mean structural features are 5 dimensions added at the end of the x_feat vector (x_feats[:, -5:]) and extract here
            if self.add_mean_structure:
                # Extract the structure tensor and resqueeze
                x_struct = x_tensor[:, 0, -5:].squeeze(1)
                x_struct = self.standardizer_structures(x_struct)
                # Extract x_tensor and preserve the windows
                x_tensor = x_tensor[:, :, :-5]

            x_tensor = self.standardizer_sequence(x_tensor)

            if x_feats is not None:
                x_feats = self.standardizer_features(self.reshape_features(x_tensor, x_feats))
                x_tensor = torch.cat([x_tensor, x_feats], dim=2)
        # First stage : standard NNAlign, Layer -> BN -> DO -> ReLU
        z = self.in_layer(x_tensor)
        # Flip dimensions to allow for batchnorm then flip back
        if self.batchnorm:
            z = self.bn1(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden)) \
                .view(-1, x_tensor.shape[1], self.n_hidden)
        z = self.dropout(z)
        z = self.act(z)
        # Additional hidden layer if used
        if self.add_hidden_layer:
            z = self.hidden_layer(z)
            if self.batchnorm:
                z = self.bnh(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden_2)) \
                    .view(-1, x_tensor.shape[1], self.n_hidden_2)
            z = self.dropout(z)
            z = self.act(z)
            z = self.out_layer(z)
        else:
            z = self.out_layer(z)  # Out Layer for prediction

        # NNAlign selecting the max score here
        with torch.no_grad():
            # Here, use sigmoid to set values to 0,1 before masking
            # only for index selection, Z will be returned as logits
            # TODO: Sigmoid here is not really necessary since we take argmax anyways?
            max_idx = torch.mul(F.sigmoid(z), x_mask).argmax(dim=1).unsqueeze(1)

        # Second stage, cat z and struture and run one more layer
        z = torch.gather(z, 1, max_idx).squeeze(1)  # Indexing the best submers ; Removed sigmoid here
        z = torch.cat([z, x_struct], dim=1)
        z = self.final_layer(z) # Here z are logits
        return z, max_idx

    def __old_forward(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_feats=None):
        with torch.no_grad():
            # TODO : Here, it's a two-stage model ; Unsafe but quick way to add the features is to split from x_feat instead of re-writing code
            #        Assumes that the mean structural features are 5 dimensions added at the end of the x_feat vector (x_feats[:, -5:]) and extract here
            if self.add_mean_structure:
                # Extract the structure tensor and resqueeze
                x_struct = x_tensor[:, 0, -5:].squeeze(1)
                x_struct = self.standardizer_structures(x_struct)
                # Extract x_tensor and preserve the windows
                x_tensor = x_tensor[:, :, :-5]

            x_tensor = self.standardizer_sequence(x_tensor)

            if x_feats is not None:
                # Take out the structural data part
                # x_struct = x_feats[:, -5:]
                # Takes the flattened x_features tensor and repeats it for each icore
                # x_feats = x_feats[:, :-5]
                x_feats = self.standardizer_features(self.reshape_features(x_tensor, x_feats))
                x_tensor = torch.cat([x_tensor, x_feats], dim=2)

        z = self.in_layer(x_tensor)
        # Flip dimensions to allow for batchnorm then flip back
        if self.batchnorm:
            z = self.bn1(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden)) \
                .view(-1, x_tensor.shape[1], self.n_hidden)
        z = self.dropout(z)
        z = self.act(z)
        # Additional hidden layer
        if self.add_hidden_layer:
            z = self.hidden_layer(z)
            if self.batchnorm:
                z = self.bnh(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden_2)) \
                    .view(-1, x_tensor.shape[1], self.n_hidden_2)
            z = self.dropout(z)
            z = self.act(z)
            z = self.out_layer(z)
        else:
            z = self.out_layer(z)  # Out Layer for prediction

        # NNAlign selecting the max score here
        with torch.no_grad():
            # Here, use sigmoid to set values to 0,1 before masking
            # only for index selection, Z will be returned as logits
            # TODO: Sigmoid here is not really necessary since we take argmax anyways?
            max_idx = torch.mul(F.sigmoid(z), x_mask).argmax(dim=1).unsqueeze(1)

        z = torch.gather(z, 1, max_idx).squeeze(1)  # Indexing the best submers ; Removed sigmoid here
        z = torch.cat([z, x_struct], dim=1)
        z = F.sigmoid(self.final_layer(
            z))  # Moved the sigmoid to here ; TODO : Check if it should be sigmoid+MSELoss or logits + BCE loss??
        return z

# OLD MODELS TO IGNORE; Here for compatibility reasons for the moment.


class NNAlign(NetParent):
    """
    This simply combines standardizer with nnalign in a slightly different architecture than EFsinglePass
    """

    def __init__(self, n_hidden, window_size, activation=nn.SELU(), batchnorm=False, dropout=0.0,
                 standardize=True, **kwargs):
        super(NNAlign, self).__init__()
        self.nnalign = NNAlignSinglePass(n_hidden, window_size, activation, batchnorm, dropout)
        self.standardizer = StandardizerSequence(window_size * 20) if standardize else StdBypass()
        # Save here to make reloading a model potentially easier
        self.init_params = {'n_hidden': n_hidden, 'window_size': window_size, 'activation': activation,
                            'batchnorm': batchnorm, 'dropout': dropout,
                            'standardize': standardize}

    def fit_standardizer(self, x_tensor: torch.Tensor, x_mask):
        assert self.training, 'Must be in training mode to fit!'
        with torch.no_grad():
            self.standardizer.fit(x_tensor, x_mask)

    def forward(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        with torch.no_grad():
            x_tensor = self.standardizer(x_tensor)
        x_tensor = self.nnalign(x_tensor, x_mask)
        return x_tensor

    def predict(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        with torch.no_grad():
            x_tensor = self.standardizer(x_tensor)
            x_tensor, max_idx = self.nnalign.predict(x_tensor, x_mask)
            return x_tensor, max_idx

    def predict_logits(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        with torch.no_grad():
            x_tensor = self.standardizer(x_tensor)
            x_tensor, max_idx = self.nnalign.predict_logits(x_tensor, x_mask)
            return x_tensor, max_idx

    def reset_parameters(self, **kwargs):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                try:
                    child.reset_parameters(**kwargs)
                except:
                    print('debug HERE', child)


class ExtraLayerSingle(NetParent):

    def __init__(self, n_input, n_hidden, activation=nn.SELU(), batchnorm=False, dropout=0.0):
        super(ExtraLayerSingle, self).__init__()
        self.n_input = n_input

        # This here exists for compatibility issues. We don't actually use any hidden but a single in -> out layer
        self.n_hidden = n_hidden
        # These here are used to batchnorm and dropout the concatenated inputs, rather than an intermediate layer nodes
        self.dropout = nn.Dropout(dropout)
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(n_input)
        self.batchnorm = batchnorm
        # Also exists for compatibility...
        self.act = activation
        self.layer = nn.Linear(n_input, 1)

    def forward(self, x_concat):
        """ Assumes we give it the concatenated (dim=1) input
        The input should be the concat'd tensor between the tensor logits returned by NNAlign and the standardized features
        Args:
            x_concat:

        Returns:
            z
        """
        if self.batchnorm:
            x_concat = self.bn1(x_concat)
        z = self.dropout(x_concat)
        z = self.layer(z)
        return z

    def predict(self, x_concat):
        """
        Convoluted but exists for compatibility issues
        Args:
            x_concat:

        Returns:

        """
        return F.sigmoid(self(x_concat))

    # def state_dict(self, **kwargs):
    #     state_dict = super(ExtraLayerSingle, self).state_dict()
    #     state_dict['n_input'] = self.n_input
    #     state_dict['n_hidden'] = self.n_hidden
    #     # state_dict['dropout'] = self.dropout.p
    #     state_dict['batchnorm'] = self.batchnorm
    #     state_dict['act'] = self.act
    #     return state_dict
    #
    # def load_state_dict(self, state_dict, **kwargs):
    #     self.n_input = state_dict['n_input']
    #     self.n_hidden = state_dict['n_hidden']
    #     # self.dropout = state_dict['dropout']
    #     self.batchnorm = state_dict['batchnorm']
    #     self.act = state_dict['act']


class ExtraLayerDouble(NetParent):
    def __init__(self, n_input, n_hidden, activation=nn.SELU(), batchnorm=False, dropout=0.0):
        super(ExtraLayerDouble, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.act = activation
        self.batchnorm = batchnorm
        self.dropout = nn.Dropout(dropout)
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(n_hidden)

        self.in_layer = nn.Linear(self.n_input, n_hidden)
        self.out_layer = nn.Linear(n_hidden, 1)

    def forward(self, x_concat):
        """ Assumes x_concat comes from the concatenation of the output of NNAlign and X_features, standardized or not
        Args:
            x_concat:

        Returns:
            z: The result of the layers
        """
        z = self.in_layer(x_concat)
        if self.batchnorm:
            z = self.bn1(z)
        z = self.act(self.dropout(z))
        z = self.out_layer(z)
        return z

    def predict(self, x_concat):
        """ Exists for compatibility / cleaner code issues
        Args:
            x_concat: Same as above.
        Returns:
            z
        """
        return F.sigmoid(self(x_concat))

    # def state_dict(self, **kwargs):
    #     state_dict = super(ExtraLayerDouble, self).state_dict()
    #     state_dict['n_input'] = self.n_input
    #     state_dict['n_hidden'] = self.n_hidden
    #     # state_dict['dropout'] = self.dropout.p
    #     state_dict['batchnorm'] = self.batchnorm
    #     state_dict['act'] = self.act
    #     return state_dict
    #
    # def load_state_dict(self, state_dict, **kwargs):
    #     self.n_input = state_dict['n_input']
    #     self.n_hidden = state_dict['n_hidden']
    #     # self.dropout = state_dict['dropout']
    #     self.batchnorm = state_dict['batchnorm']
    #     self.act = state_dict['act']


class NNAlignEF_OLD(NetParent):
    """ EF == ExtraFeatures
    TODO: Currently assumes that I need an extra in_layer + an extra out_layer
          This does not use ExtraLayerSingle/Double, that's in EF2, but this is an easier way if we don't need those classes
          Could also be changed to take a single extra layer of nn.Linear(1+n_extrafeatures, 1)
          That takes as input the logits from NNAlign + the extra features and directly returns a score without 2 layers.
          Can maybe write another class EFModel that just takes the ef_xx part here

    """

    def __init__(self, n_hidden, window_size, activation=nn.SELU(), batchnorm=False, dropout=0.0, standardize=True,
                 n_extrafeatures=0, n_hidden_ef=5, activation_ef=nn.SELU(), batchnorm_ef=False, dropout_ef=0.0,
                 **kwargs):
        super(NNAlignEF_OLD, self).__init__()
        # NNAlign part
        self.nnalign_model = NNAlign(n_hidden, window_size, activation, batchnorm, dropout, standardize)
        # Extra layer part
        self.in_dim = n_extrafeatures + 1  # +1 because that's the dimension of the logit scores returned by NNAlign
        self.ef_standardizer = StandardizerFeatures(n_feats=n_extrafeatures) if standardize else StdBypass()
        self.ef_inlayer = nn.Linear(self.in_dim, n_hidden_ef)
        self.ef_outlayer = nn.Linear(n_hidden_ef, 1)
        self.ef_act = activation_ef
        self.ef_dropout = nn.Dropout(dropout_ef)
        self.ef_batchnorm = batchnorm_ef

        # TODO : If this is switched to a single layer, then BatchNorm1d should be updated to nn.BatchNorm1d(self.in_dim)
        if batchnorm_ef:
            self.ef_bn1 = nn.BatchNorm1d(n_hidden_ef)

        self.init_params = {'n_hidden': n_hidden, 'window_size': window_size, 'activation': activation,
                            'batchnorm': batchnorm, 'dropout': dropout, 'standardize': standardize,
                            'n_extrafeatures': n_extrafeatures, 'n_hidden_ef': n_hidden_ef,
                            'activation_ef': activation_ef,
                            'batchnorm_ef': batchnorm_ef, 'dropout_ef': dropout_ef}

    def fit_standardizer(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_features: torch.Tensor):
        self.nnalign_model.fit_standardizer(x_tensor, x_mask)
        self.ef_standardizer.fit(x_features)

    def forward(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_features: torch.Tensor):
        # NNAlign part
        z = self.nnalign_model(x_tensor, x_mask)
        # Extra features part, standardizes, concat
        x_features = self.ef_standardizer(x_features)
        z = torch.cat([z, x_features], dim=1)
        # Standard NN stuff for the extra layers
        z = self.ef_inlayer(z)
        if self.ef_batchnorm:
            z = self.ef_bn1(z)
        z = self.ef_act(self.ef_dropout(z))
        # Returning logits
        z = self.ef_outlayer(z)
        return z

    def predict(self, x_tensor: torch.Tensor, x_mask: torch.Tensor, x_features: torch.Tensor):
        """ TODO: This is a bit convoluted and could be reworked to be more efficient
                  Would probly require to modify the other classes a bit though

        Args:
            x_tensor:
            x_mask:
            x_features:

        Returns:

        """
        with torch.no_grad():
            # Return logits from nnalign model + max idx
            z, max_idx = self.nnalign_model.predict_logits(x_tensor, x_mask)

            # Standard NN stuff for the extra layers
            x_features = self.ef_standardizer(x_features)
            z = torch.cat([z, x_features], dim=1)
            z = self.ef_inlayer(z)
            if self.ef_batchnorm:
                z = self.ef_bn1(z)
            z = self.ef_act(self.ef_dropout(z))
            # Returning probs [0, 1]
            z = F.sigmoid(self.ef_outlayer(z))
            return z, max_idx
