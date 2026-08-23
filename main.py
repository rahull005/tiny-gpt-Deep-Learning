import os

try:
    import numpy as np
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing required dependencies. Install them with:\n"
        "  python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt"
    ) from exc


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "data/input.txt"

BLOCK_SIZE = 128
BATCH_SIZE = 32

N_EMBD = 128
N_HEAD = 4
N_LAYER = 4

LEARNING_RATE = 3e-4

MAX_STEPS = 5000
EVAL_INTERVAL = 500

TEMPERATURE = 0.8

MODEL_PATH = "tiny_gpt.weights.h5"


# ============================================================
# 2. RANDOM SEEDS
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 3. LOAD DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Training file not found: {DATA_PATH}"
    )

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

if len(text) < BLOCK_SIZE + 2:
    raise ValueError(
        "Training text is too small for the selected BLOCK_SIZE."
    )

print("=" * 60)
print("DATASET")
print("=" * 60)

print(f"Characters in dataset : {len(text)}")


# ============================================================
# 4. CHARACTER TOKENIZER
# ============================================================

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = {
    ch: i
    for i, ch in enumerate(chars)
}

itos = {
    i: ch
    for i, ch in enumerate(chars)
}


def encode(s):
    return [
        stoi[c]
        for c in s
    ]


def decode(ids):
    return "".join(
        itos[int(i)]
        for i in ids
    )


print(f"Vocabulary size       : {vocab_size}")
print(f"Characters             : {repr(''.join(chars))}")


# ============================================================
# 5. CONVERT TEXT TO TOKEN IDs
# ============================================================

data = np.array(
    encode(text),
    dtype=np.int32
)


# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

print(f"Training characters    : {len(train_data)}")
print(f"Validation characters  : {len(val_data)}")


# ============================================================
# 7. BATCH CREATION
# ============================================================

def get_batch(split):
    """
    Creates a batch of input/target sequences.

    Example:

        x = abcde
        y = bcdef

    The model learns next-token prediction.
    """

    dataset = (
        train_data
        if split == "train"
        else val_data
    )

    max_start = len(dataset) - BLOCK_SIZE - 1

    if max_start <= 0:
        raise ValueError(
            f"{split} dataset is too small for BLOCK_SIZE={BLOCK_SIZE}"
        )

    ix = np.random.randint(
        0,
        max_start,
        size=BATCH_SIZE
    )

    x = np.stack([
        dataset[i:i + BLOCK_SIZE]
        for i in ix
    ])

    y = np.stack([
        dataset[i + 1:i + BLOCK_SIZE + 1]
        for i in ix
    ])

    return (
        tf.convert_to_tensor(x, dtype=tf.int32),
        tf.convert_to_tensor(y, dtype=tf.int32)
    )


# ============================================================
# 8. MULTI-HEAD SELF-ATTENTION
# ============================================================

class MultiHeadSelfAttention(tf.keras.layers.Layer):

    def __init__(
        self,
        n_embd,
        n_head
    ):
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError(
                "n_embd must be divisible by n_head."
            )

        self.n_head = n_head

        self.head_size = n_embd // n_head

        # Q projection
        self.Wq = tf.keras.layers.Dense(
            n_embd,
            use_bias=False
        )

        # K projection
        self.Wk = tf.keras.layers.Dense(
            n_embd,
            use_bias=False
        )

        # V projection
        self.Wv = tf.keras.layers.Dense(
            n_embd,
            use_bias=False
        )

        # Output projection
        self.Wo = tf.keras.layers.Dense(
            n_embd,
            use_bias=False
        )

    def split_heads(self, x):

        """
        Input:

            (B, T, C)

        Output:

            (B, H, T, HS)
        """

        shape = tf.shape(x)

        B = shape[0]
        T = shape[1]

        x = tf.reshape(
            x,
            (
                B,
                T,
                self.n_head,
                self.head_size
            )
        )

        x = tf.transpose(
            x,
            perm=[0, 2, 1, 3]
        )

        return x

    def call(self, x):

        # ----------------------------------------------------
        # Shapes
        # ----------------------------------------------------

        shape = tf.shape(x)

        B = shape[0]
        T = shape[1]

        C = x.shape[-1]

        # ----------------------------------------------------
        # Q, K, V
        # ----------------------------------------------------

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # ----------------------------------------------------
        # Attention scores
        #
        # Q @ K^T
        # ----------------------------------------------------

        scores = tf.matmul(
            Q,
            K,
            transpose_b=True
        )

        # ----------------------------------------------------
        # Scaling
        #
        # divide by sqrt(head_size)
        # ----------------------------------------------------

        scale = tf.sqrt(
            tf.cast(
                self.head_size,
                tf.float32
            )
        )

        scores = scores / scale

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        mask = tf.linalg.band_part(
            tf.ones(
                (T, T),
                dtype=tf.float32
            ),
            -1,
            0
        )

        # Future positions become -infinity
        scores = tf.where(
            mask[None, None, :, :] == 0,
            tf.constant(
                -1e9,
                dtype=tf.float32
            ),
            scores
        )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        attention_weights = tf.nn.softmax(
            scores,
            axis=-1
        )

        # ----------------------------------------------------
        # Weighted sum of V
        # ----------------------------------------------------

        head_output = tf.matmul(
            attention_weights,
            V
        )

        # ----------------------------------------------------
        # Merge heads
        #
        # (B,H,T,HS)
        #
        # ->
        #
        # (B,T,H,HS)
        # ----------------------------------------------------

        head_output = tf.transpose(
            head_output,
            perm=[0, 2, 1, 3]
        )

        # ----------------------------------------------------
        # Concatenate heads
        # ----------------------------------------------------

        merged = tf.reshape(
            head_output,
            (
                B,
                T,
                C
            )
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        return self.Wo(merged)


# ============================================================
# 9. FEED-FORWARD NETWORK
# ============================================================

class FeedForward(tf.keras.layers.Layer):

    def __init__(self, n_embd):

        super().__init__()

        self.net = tf.keras.Sequential([

            tf.keras.layers.Dense(
                4 * n_embd,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                n_embd
            )
        ])

    def call(self, x):

        return self.net(x)


# ============================================================
# 10. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        n_embd,
        n_head
    ):

        super().__init__()

        # Pre-normalization
        self.ln1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-5
        )

        self.attention = MultiHeadSelfAttention(
            n_embd,
            n_head
        )

        self.ln2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-5
        )

        self.feed_forward = FeedForward(
            n_embd
        )

    def call(self, x):

        # ----------------------------------------------------
        # Attention sub-layer
        #
        # x -> LayerNorm -> Attention -> Residual
        # ----------------------------------------------------

        x = (
            x
            + self.attention(
                self.ln1(x)
            )
        )

        # ----------------------------------------------------
        # Feed-forward sub-layer
        #
        # x -> LayerNorm -> FFN -> Residual
        # ----------------------------------------------------

        x = (
            x
            + self.feed_forward(
                self.ln2(x)
            )
        )

        return x


# ============================================================
# 11. TINY GPT
# ============================================================

class TinyGPT(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd,
        n_head,
        n_layer
    ):

        super().__init__()

        self.block_size = block_size

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = tf.keras.layers.Embedding(
            vocab_size,
            n_embd
        )

        # ----------------------------------------------------
        # Position embedding
        # ----------------------------------------------------

        self.position_embedding = tf.keras.layers.Embedding(
            block_size,
            n_embd
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = [

            TransformerBlock(
                n_embd,
                n_head
            )

            for _ in range(n_layer)
        ]

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.final_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-5
        )

        # ----------------------------------------------------
        # Language model head
        # ----------------------------------------------------

        self.lm_head = tf.keras.layers.Dense(
            vocab_size
        )

    def call(self, idx):

        # ----------------------------------------------------
        # Sequence length
        # ----------------------------------------------------

        T = tf.shape(idx)[1]

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        token_embeddings = self.token_embedding(
            idx
        )

        # ----------------------------------------------------
        # Position embeddings
        # ----------------------------------------------------

        positions = tf.range(T)

        position_embeddings = self.position_embedding(
            positions
        )

        # ----------------------------------------------------
        # Combine token + position
        # ----------------------------------------------------

        x = (
            token_embeddings
            + position_embeddings
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        x = self.final_ln(x)

        # ----------------------------------------------------
        # Convert representation to vocabulary logits
        # ----------------------------------------------------

        logits = self.lm_head(x)

        return logits


# ============================================================
# 12. CREATE MODEL
# ============================================================

model = TinyGPT(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER
)


# ============================================================
# 13. BUILD MODEL
# ============================================================

dummy_input = tf.zeros(
    (
        1,
        BLOCK_SIZE
    ),
    dtype=tf.int32
)

_ = model(dummy_input)


# ============================================================
# 14. MODEL SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL")
print("=" * 60)

model.summary()


# ============================================================
# 15. LOSS FUNCTION
# ============================================================

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True
)


# ============================================================
# 16. OPTIMIZER
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


# ============================================================
# 17. TRAINING STEP
# ============================================================

@tf.function
def train_step(xb, yb):

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        logits = model(
            xb,
            training=True
        )

        # ----------------------------------------------------
        # Cross entropy
        # ----------------------------------------------------

        loss = loss_fn(
            yb,
            logits
        )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    gradients = tape.gradient(
        loss,
        model.trainable_variables
    )

    # --------------------------------------------------------
    # Update weights
    # --------------------------------------------------------

    optimizer.apply_gradients(
        zip(
            gradients,
            model.trainable_variables
        )
    )

    return loss


# ============================================================
# 18. EVALUATION
# ============================================================

def evaluate():

    # We don't need gradients while evaluating.

    xb, yb = get_batch("train")

    train_logits = model(
        xb,
        training=False
    )

    train_loss = loss_fn(
        yb,
        train_logits
    )

    xb, yb = get_batch("val")

    val_logits = model(
        xb,
        training=False
    )

    val_loss = loss_fn(
        yb,
        val_logits
    )

    return (
        float(train_loss.numpy()),
        float(val_loss.numpy())
    )


# ============================================================
# 19. TEXT GENERATION
# ============================================================

def generate(
    model,
    start_text,
    max_new_tokens=300,
    temperature=0.8
):

    # --------------------------------------------------------
    # Encode starting text
    # --------------------------------------------------------

    ids = encode(start_text)

    idx = np.array(
        [ids],
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Generate token by token
    # --------------------------------------------------------

    for _ in range(max_new_tokens):

        # ----------------------------------------------------
        # Keep only the latest context window
        # ----------------------------------------------------

        idx_cond = idx[
            :,
            -BLOCK_SIZE:
        ]

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        logits = model(
            tf.convert_to_tensor(
                idx_cond,
                dtype=tf.int32
            ),
            training=False
        )

        # ----------------------------------------------------
        # Only the final position matters
        # ----------------------------------------------------

        logits_last = logits[
            :,
            -1,
            :
        ]

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        logits_last = (
            logits_last
            / temperature
        )

        # ----------------------------------------------------
        # Convert to probabilities
        # ----------------------------------------------------

        probabilities = tf.nn.softmax(
            logits_last,
            axis=-1
        ).numpy()

        # ----------------------------------------------------
        # Sample next token
        # ----------------------------------------------------

        next_token = np.array([
            np.random.choice(
                vocab_size,
                p=probabilities[b]
            )
            for b in range(idx.shape[0])
        ])

        next_token = next_token[:, None]

        # ----------------------------------------------------
        # Append
        # ----------------------------------------------------

        idx = np.concatenate(
            [
                idx,
                next_token
            ],
            axis=1
        )

    return decode(
        idx[0].tolist()
    )


# ============================================================
# 20. TRAINING
# ============================================================

print()
print("=" * 60)
print("TRAINING")
print("=" * 60)

for step in range(
    1,
    MAX_STEPS + 1
):

    xb, yb = get_batch(
        "train"
    )

    loss = train_step(
        xb,
        yb
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if (
        step == 1
        or step % EVAL_INTERVAL == 0
    ):

        train_loss, val_loss = evaluate()

        print(
            f"step {step:5d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f}"
        )


# ============================================================
# 21. SAVE MODEL
# ============================================================

model.save_weights(
    MODEL_PATH
)

print()
print("=" * 60)
print(f"Model saved to: {MODEL_PATH}")
print("=" * 60)


# ============================================================
# 22. GENERATE TEXT
# ============================================================

print()
print("=" * 60)
print("GENERATED TEXT")
print("=" * 60)

generated_text = generate(
    model,
    start_text="to be",
    max_new_tokens=500,
    temperature=TEMPERATURE
)

print()
print(generated_text)