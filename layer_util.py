from torch import nn
import torch.nn.functional as F


def get_image_layer(name, rank):
  """Get an N-D layer object.

  Args:
    name: A `str`. The name of the requested layer.
    rank: An `int`. The rank of the requested layer.

  Returns:
    A `torch.nn.Module` object.

  Raises:
    ValueError: If the requested layer is unknown.
  """
  try:
    return _IMAGE_LAYERS[(name, rank)]
  except KeyError as err:
    raise ValueError(
        f"Could not find a layer with name '{name}' and rank {rank}.") from err
  
  
# def get_graph_layer(name):
#   """Get an graph layer object.

#   Args:
#     name: A `str`. The name of the requested layer.

#   Returns:
#     A `torch.nn.Module` object.

#   Raises:
#     ValueError: If the requested layer is unknown.
#   """
#   try:
#     return _GRAPH_LAYERS[name] 
#   except KeyError as err:
#     raise ValueError(
#         f"Could not find an activation with name '{name}'") from err
  
# def get_default_kwargs(name):
#   """Get an graph layer object.

#   Args:
#     name: A `str`. The name of the requested layer.

#   Returns:
#     A `torch.nn.Module` object.

#   Raises:
#     ValueError: If the requested layer is unknown.
#   """
#   try:
#     return _LAYER_KWARGS[name] 
#   except KeyError as err:
#     raise ValueError(
#         f"Could not find an activation with name '{name}'") from err
  
  
def get_activation(name):
  """Get an activation object.

  Args:
    name: A `str`. The name of the requested layer.

  Returns:
    A `torch.nn.Module` object.

  Raises:
    ValueError: If the requested activation is unknown.
  """
  try:
    return _ACTIVATIONS[name]
  except KeyError as err:
    raise ValueError(
        f"Could not find an activation with name '{name}'") from err


class TimeDistributed(nn.Module):
    """
    Apply a module independently to each time step.

    Expected input shape:
        (N, C, T, D, H, W)
         |  |  |  └──── spatial dims
         |  |  └────── time
         |  └───────── channels
         └──────────── batch

    The wrapped module should accept:
        (N, C, D, H, W)

    Output shape:
        (N, C_out, T, D_out, H_out, W_out)
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        # Original shape: (N, C, T, D, H, W)
        N, C, T, D, H, W = x.shape

        # Step 1: Move time next to batch so we can merge them
        # (N, C, T, D, H, W) -> (N, T, C, D, H, W)
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()

        # Step 2: Merge batch and time into a single batch dimension
        # (N, T, C, D, H, W) -> (N*T, C, D, H, W)
        x = x.reshape(N * T, C, D, H, W)

        # Step 3: Apply the module to each time slice independently
        # Module sees a standard 5D tensor batch
        y = self.module(x)  # (N*T, C_out, D_out, H_out, W_out)

        # Extract new shape after module
        _, C_out, D_out, H_out, W_out = y.shape

        # Step 4: Restore time dimension
        # (N*T, C_out, D_out, H_out, W_out) -> (N, T, C_out, D_out, H_out, W_out)
        y = y.reshape(N, T, C_out, D_out, H_out, W_out)

        # Step 5: Move channels back to expected position
        # (N, T, C_out, D_out, H_out, W_out) -> (N, C_out, T, D_out, H_out, W_out)
        y = y.permute(0, 2, 1, 3, 4, 5).contiguous()

        return y

class SpaceDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        # Original shape: (N, C, T, D, H, W)
        N, C, T, D, H, W = x.shape

        # --- Step 1: Move spatial axes next to batch ---
        # (N, C, T, D, H, W) -> (N, D, H, W, C, T)
        x = x.permute(0, 3, 4, 5, 1, 2).contiguous()

        # --- Step 2: Flatten N * D * H * W into a single batch axis ---
        # (N, D, H, W, C, T) -> (N*D*H*W, C, T)
        x = x.reshape(N * D * H * W, C, T)

        # --- Step 3: Apply module independently per spatial location ---
        y = self.module(x)

        _, C_out, T_out = y.shape

        # --- Step 4: Restore location ---
        y = y.reshape(N, D, H, W, C_out, T_out)

        # Step 5: Move channels back to expected position
        # (N, D, H, W, C_out, T_out) -> (N, C_out, T_out, D, H, W)
        y = y.permute(0, 4, 5, 1, 2, 3).contiguous()
        return y



class Conv4d(nn.Module):
    def __init__(self,
                in_channels, 
                out_channels,
                kernel_size, 
                stride=1,
                padding=0, 
                dilation=1, 
                groups=1, 
                bias=True, 
                padding_mode='zeros', 
                device=None, 
                dtype=None):
        super().__init__()
        # split kernel, stride, padding into temporal and spatial components
        self.time_kernel, self.space_kernel = self._split_param(kernel_size, "kernel_size")
        self.time_stride, self.space_stride = self._split_param(stride, "stride")
        self.time_padding, self.space_padding = self._split_param(padding, "padding")

        # Spatial 3D conv per time step
        self.conv3d = TimeDistributed(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=self.space_kernel,
                stride=self.space_stride,
                padding=self.space_padding
            )
        )

        # Temporal 1D conv across time dimension
        self.conv1d = SpaceDistributed(
           nn.Conv1d(
              out_channels,
              out_channels,
              kernel_size=self.time_kernel,
              stride=self.time_stride,
              padding=self.time_padding
            )
        )

    def _split_param(self, param, name="parameter"):
      """Split a 4D param into (time, spatial_tuple)"""
      if isinstance(param, (tuple, list)):
          if len(param) != 4:
              raise ValueError(f"{name} must have 4 elements (time, D, H, W)")
          return param[0], tuple(param[1:])
      else:
          return param, (param,) * 3

    def forward(self, x):
        # x: (B, C, T, D, H, W)

        # ---- Spatial conv per time ----
        x = self.conv3d(x)
        # Apply voxel-wise Conv1D over time
        x = self.conv1d(x)
        return x


class BatchNorm4d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        # Spatial BatchNorm3d per time step
        self.bn3d = TimeDistributed(
            nn.BatchNorm3d(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats
            )
        )
        # Temporal BatchNorm1d voxel-wise
        self.bn1d = SpaceDistributed(
            nn.BatchNorm1d(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats
            )
        )
    def forward(self, x):
        # x: (B, C, T, D, H, W)

        # ---- Spatial conv per time ----
        x = self.bn3d(x)
        # Apply voxel-wise Conv1D over time
        x = self.bn1d(x)
        return x


class Upsample4d(nn.Module):
    """
    Separable 4D upsampling:
        - Spatial upsampling per time step (3D)
        - Temporal upsampling per voxel (1D)

    Expected input:
        (B, C, T, D, H, W)

    Output:
        (B, C, T_out, D_out, H_out, W_out)
    """
    def __init__(
        self,
        scale_factor=None,
        size=None,
        mode="trilinear",
        align_corners=False,
    ):
        super().__init__()

        if scale_factor is None and size is None:
            raise ValueError("Either scale_factor or size must be provided")

        # Split parameters into temporal + spatial
        self.time_scale, self.space_scale = self._split_param(scale_factor, size)

        # ---- Spatial upsampling (per time step) ----
        # Uses 3D interpolation over (D, H, W)
        if self.space_scale is not None:
            self.up3d = TimeDistributed(
                nn.Upsample(
                    scale_factor=self.space_scale,
                    mode=mode,
                    align_corners=align_corners if "linear" in mode else None,
                )
            )
        else:
            self.up3d = TimeDistributed(
                nn.Upsample(
                    size=self.space_scale,
                    mode=mode,
                    align_corners=align_corners if "linear" in mode else None,
                )
            )

        # ---- Temporal upsampling (per voxel) ----
        # Uses 1D interpolation over T
        if self.time_scale is not None:
            self.up1d = SpaceDistributed(
                nn.Upsample(
                    scale_factor=self.time_scale,
                    mode="linear",
                    align_corners=align_corners,
                )
            )
        else:
            self.up1d = SpaceDistributed(
                nn.Upsample(
                    size=self.time_scale,
                    mode="linear",
                    align_corners=align_corners,
                )
            )

    def _split_param(self, scale_factor, size):
        """
        Split 4D (T, D, H, W) into:
        - time scalar
        - spatial tuple (D, H, W)
        """
        if scale_factor is not None:
            if isinstance(scale_factor, (tuple, list)):
                if len(scale_factor) != 4:
                    raise ValueError("scale_factor must be (T, D, H, W)")
                return scale_factor[0], tuple(scale_factor[1:])
            else:
                return scale_factor, (scale_factor,) * 3

        if size is not None:
            if isinstance(size, (tuple, list)):
                if len(size) != 4:
                    raise ValueError("size must be (T, D, H, W)")
                return size[0], tuple(size[1:])
            else:
                return size, (size,) * 3

        return None, None

    def forward(self, x):
        # x: (B, C, T, D, H, W)

        # 1) Spatial upsample per time slice
        x = self.up3d(x)   # (B, C, T, D*, H*, W*)

        # 2) Temporal upsample per voxel
        x = self.up1d(x)   # (B, C, T*, D*, H*, W*)

        return x


class MaxPool4d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        # split kernel, stride, padding into temporal and spatial components
        self.time_kernel, self.space_kernel = self._split_param(kernel_size, "kernel_size")
        self.time_stride, self.space_stride = self._split_param(stride if stride is not None else kernel_size, "stride")
        self.time_padding, self.space_padding = self._split_param(padding, "padding")

        # Spatial 3D pooling per time slice
        self.pool3d = TimeDistributed(
            nn.MaxPool3d(
                kernel_size=self.space_kernel,
                stride=self.space_stride,
                padding=self.space_padding
            )
        )

        # Temporal 1D pooling across time, voxel-wise
        self.pool1d = SpaceDistributed(
            nn.MaxPool1d(
                kernel_size=self.time_kernel,
                stride=self.time_stride,
                padding=self.time_padding
            )
        )

    def _split_param(self, param, name="parameter"):
        if isinstance(param, (tuple, list)):
            if len(param) != 4:
                raise ValueError(f"{name} must have 4 elements (time, D, H, W)")
            return param[0], tuple(param[1:])
        else:
            return param, (param,) * 3

    def forward(self, x):
        # x: (B, C, T, D, H, W)

        # ---- Spatial conv per time ----
        x = self.pool3d(x)
        # Apply voxel-wise Conv1D over time
        x = self.pool1d(x)
        return x



_IMAGE_LAYERS = {
    ('AveragePooling', 1): nn.AvgPool1d,
    ('AveragePooling', 2): nn.AvgPool2d,
    ('AveragePooling', 3): nn.AvgPool3d,
    ('Conv', 1): nn.Conv1d,
    ('Conv', 2): nn.Conv2d,
    ('Conv', 3): nn.Conv3d,
    ('Conv', 4): Conv4d,
    ('ConvTranspose', 1): nn.ConvTranspose1d,
    ('ConvTranspose', 2): nn.ConvTranspose2d,
    ('ConvTranspose', 3): nn.ConvTranspose3d,
    ('MaxPool', 1): nn.MaxPool1d,
    ('MaxPool', 2): nn.MaxPool2d,
    ('MaxPool', 3): nn.MaxPool3d,
    ('MaxPool', 4): MaxPool4d,
    ('Dropout', 1): nn.Dropout1d,
    ('Dropout', 2): nn.Dropout2d,
    ('Dropout', 3): nn.Dropout3d,
    ('ZeroPadding', 1): nn.ZeroPad1d,
    ('ZeroPadding', 2): nn.ZeroPad2d,
    ('ZeroPadding', 3): nn.ZeroPad3d,
    ('BatchNorm', 1): nn.BatchNorm1d,
    ('BatchNorm', 2): nn.BatchNorm2d,
    ('BatchNorm', 3): nn.BatchNorm3d,
    ('BatchNorm', 4): BatchNorm4d,
    ('InstanceNorm', 1): nn.InstanceNorm1d,
    ('InstanceNorm', 2): nn.InstanceNorm2d,
    ('InstanceNorm', 3): nn.InstanceNorm3d
}

# _GRAPH_LAYERS = {
#   'ChebConv': gnn.ChebConv,
#   'GraphConv': gnn.GraphConv,
#   'GCNConv': gnn.GCNConv,
#   'GATConv': gnn.GATConv,
#   'InstanceNorm': gnn.InstanceNorm,
#   'BatchNorm': gnn.BatchNorm,
#   'GraphNorm': gnn.GraphNorm
# }

# _LAYER_KWARGS = {
#   'ChebConv': {'K':3},
#   'GraphConv': {},
#   'GCNConv': {},
#   'GATConv': {}
# }


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
    "softmax": nn.Softmax
}


# def init_weights(m):
#     if isinstance(m, gnn.ChebConv):
#         for lin in m.lins:
#             nn.init.kaiming_normal_(lin.weight, nonlinearity='leaky_relu')
#             lin.weight.data *= 0.1
#             if lin.bias is not None:
#                 nn.init.zeros_(lin.bias)