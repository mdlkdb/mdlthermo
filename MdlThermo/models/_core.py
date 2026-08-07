import keras
from keras.ops import numpy as onp


class ReadOut(keras.layers.Layer):
    """Apply sum pooling across the node axis.

    Args:
        name: Optional Keras layer name.
    """

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)

    def call(self, x):
        """Pool node features into graph-level features.

        Args:
            x: Node feature tensor with shape
                ``(..., num_nodes, num_features)``.

        Returns:
            A tensor with shape ``(..., num_features)``.
        """

        return onp.sum(x, axis=-2)


class GraphConvolution(keras.layers.Layer):
    """Apply an adjacency-weighted dense graph convolution.

    The layer computes ``LeakyReLU(Dense(efm @ nfm))``.
    Callers are responsible for adding self-loops or normalizing ``efm``
    when required.

    Args:
        dim: Number of output features per node.
        name: Optional Keras layer name.
    """

    def __init__(self, dim: int, *, name: str | None = None) -> None:
        super().__init__(name=name)

        self.dim = dim
        self.dot = keras.layers.Dot(axes=(2, 1))
        self.dense = keras.layers.Dense(
            dim,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
        )
        self.activation = keras.layers.LeakyReLU()

    def call(self, nfm, efm):
        """Convolve node features with an edge matrix.

        Args:
            nfm: Node feature tensor with shape
                ``(batch_size, input_nodes, input_dim)``.
            efm: Edge matrix tensor with shape
                ``(batch_size, output_nodes, input_nodes)``.

        Returns:
            A tensor with shape ``(batch_size, output_nodes, dim)``.
        """

        x = self.dot([efm, nfm])
        x = self.dense(x)

        return self.activation(x)


def GCGCN(*, node_feature_dim: int, graph_dim: int, hidden_dim: int) -> keras.Model:
    """Build a group-contribution graph convolutional network.

    The architecture applies two graph convolution layers, sum pooling,
    three hidden dense layers, and a scalar output layer.

    Args:
        graph_dim: Number of output features in each graph convolution
            layer.
        hidden_dim: Number of units in each hidden dense layer.

    Returns:
        A two-input Keras model whose output has one value per graph.
    """

    # Inputs
    nfm = keras.Input(name="Node Feature Matrix", shape=(None, node_feature_dim))
    efm = keras.Input(name="Edge Feature Matrix", shape=(None, None))

    # Graph Convolution Network
    g1 = GraphConvolution(graph_dim, name="Graph Convolution 1")(nfm, efm)
    g2 = GraphConvolution(graph_dim, name="Graph Convolution 2")(g1, efm)

    # Concatenate
    x = keras.layers.Concatenate(axis=-1)([nfm, g1, g2])

    # Summation Pooling
    x = ReadOut()(x)

    # MLP
    x = keras.layers.Dense(
        hidden_dim,
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        activation="leaky_relu",
        name="Dense 1",
    )(x)
    x = keras.layers.Dense(
        hidden_dim,
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        activation="leaky_relu",
        name="Dense 2",
    )(x)
    x = keras.layers.Dense(
        hidden_dim,
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        activation="leaky_relu",
        name="Dense 3",
    )(x)
    x = keras.layers.Dense(
        1,
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        name="Ouput",
    )(x)

    model = keras.Model(inputs=[nfm, efm], outputs=x)
    model.summary()

    return model
