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

DROPOUT_RATE = 0.1   # fraction of activations randomly zeroed during training (see note below)

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

# Sanity check to keep in mind while training: a completely untrained model,
# guessing uniformly at random over `vocab_size` classes, has expected
# cross-entropy loss = -log(1/vocab_size) = log(vocab_size).
# e.g. for vocab_size=65, that's ln(65) ~= 4.17.
# Your very first printed training loss should land close to this number —
# if it doesn't, something in the loss/logits wiring is likely broken.
print(f"Expected loss at init (ln(vocab_size)) : {np.log(vocab_size):.4f}")


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

    The model learns next-token prediction: at every position t,
    x[t] is context and y[t] is the correct next character. Every
    position in the block is trained simultaneously (not one at a time),
    which is what makes transformer training parallelizable.
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

        # Q projection: "what is this token looking for in other tokens?"
        self.Wq = tf.keras.layers.Dense(
            n_embd,
            use_bias=False
        )

        # K projection: "what does this token contain/advertise?"
        # Q and K live in the same space so their dot product is a
        # meaningful similarity/relevance score.
        self.Wk = tf.keras.layers.Dense(
            n_embd,
            use_bias=False
        )

        # V projection: "what information does this token actually pass on
        # if it gets picked?" — deliberately a SEPARATE space from Q/K,
        # because "how relevant am I" and "what do I contribute" are
        # different questions.
        self.Wv = tf.keras.layers.Dense(
            n_embd,
            use_bias=False
        )

        # Output projection: after concatenating all heads back together,
        # this lets the model mix/recombine information ACROSS heads
        # (each head only saw its own slice of the embedding until now).
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

        Splitting into H heads of size HS = C/H lets the model learn several
        INDEPENDENT relevance patterns in parallel (e.g. one head tracks
        "the previous word", another tracks "the subject of the sentence"),
        instead of being forced to learn one single attention pattern.
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
        #
        # scores[i, j] = dot product of query i with key j = how much
        # token i should attend to token j. Large positive dot product
        # -> vectors point the same direction -> "relevant".
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
        #
        # WHY: Q.K is a sum of head_size independent terms, each with some
        # variance v. The variance of the SUM grows linearly with head_size,
        # so its standard deviation grows with sqrt(head_size). Without
        # correcting for this, larger head_size -> larger-magnitude scores
        # -> softmax saturates into a near one-hot distribution -> gradients
        # through softmax vanish almost everywhere. Dividing by sqrt(head_size)
        # keeps the scores' scale roughly independent of head_size, so
        # softmax starts in its "sensitive" (well-behaved gradient) regime.
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
        #
        # A GPT predicts the next token, so token t must NEVER be allowed
        # to see tokens > t (that would be cheating — peeking at the answer
        # during training). We enforce this by setting disallowed positions'
        # score to -infinity BEFORE softmax, so softmax assigns them
        # probability e^(-inf) = 0.
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
        #
        # Converts raw scores into a probability distribution over "which
        # positions to attend to" — every row sums to exactly 1. This turns
        # attention into a weighted AVERAGE (a convex combination) of the
        # Value vectors, rather than an arbitrary linear combination.
        # ----------------------------------------------------

        attention_weights = tf.nn.softmax(
            scores,
            axis=-1
        )

        # ----------------------------------------------------
        # Weighted sum of V
        #
        # output[i] = sum_j( attention_weights[i,j] * V[j] )
        # i.e. "blend together the Values of every token, weighted by how
        # relevant each one is to token i".
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
    """
    Attention lets tokens exchange information with each other.
    FeedForward is the complementary step: each token, independently,
    "thinks about" what it just gathered. It's applied identically and
    separately to every position (no cross-token mixing happens here).

    Expanding to 4x width then projecting back down (n_embd -> 4*n_embd ->
    n_embd) gives the network a wider intermediate space to compute in —
    empirically this ~4x ratio is what the original Transformer/GPT papers
    settled on; it's a capacity/compute tradeoff, not a hard requirement.
    """

    def __init__(self, n_embd, dropout_rate=0.1):

        super().__init__()

        self.net = tf.keras.Sequential([

            tf.keras.layers.Dense(
                4 * n_embd,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                n_embd
            ),

            # ------------------------------------------------
            # DROPOUT (regularization to fight overfitting)
            #
            # MATH INTUITION: during training, each unit is independently
            # zeroed out with probability `dropout_rate`, and the surviving
            # units are scaled by 1/(1 - dropout_rate) ("inverted dropout")
            # so the expected sum of activations stays the same whether or
            # not dropout is applied. This means:
            #   E[output_with_dropout] == E[output_without_dropout]
            # so downstream layers see a signal of the same expected
            # magnitude at train and test time.
            #
            # WHY THIS FIGHTS OVERFITTING: a unit can no longer count on any
            # specific OTHER unit being present on a given forward pass
            # (it might be dropped). This discourages the network from
            # building brittle, co-adapted pathways that only work for
            # exact training examples — it's forced toward more redundant,
            # generalizable representations. This is mathematically similar
            # to training an ensemble of many thinned sub-networks and
            # implicitly averaging over them.
            #
            # At inference (training=False), dropout is simply the identity
            # function — all units are used, nothing is zeroed or rescaled.
            # ------------------------------------------------
            tf.keras.layers.Dropout(dropout_rate),
        ])

    def call(self, x, training=False):
        return self.net(x, training=training)


# ============================================================
# 10. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(tf.keras.layers.Layer):
    """
    Pre-norm residual architecture (same as GPT-2 onward):

        x = x + Dropout(Attention(LayerNorm(x)))
        x = x + FeedForward(LayerNorm(x))          (dropout lives inside FeedForward)

    RESIDUAL CONNECTION MATH INTUITION:
    For y = x + f(x), the gradient of any downstream loss L w.r.t. x is:
        dL/dx = dL/dy * dy/dx = dL/dy * (1 + df/dx)
    That "+1" term means gradient always has a direct, unattenuated path
    back to x, no matter how small or poorly-scaled df/dx is. Stack many
    such blocks and gradients can still reach the earliest layers instead
    of vanishing — this is precisely why deep transformers are trainable
    at all.

    PRE-NORM (LayerNorm INSIDE the residual branch, not after the addition)
    intuition: it keeps the "residual stream" x accumulating raw, unscaled
    updates across all layers, while each sublayer gets a nicely-normalized
    view of x to compute from. Post-norm (original 2017 Transformer paper)
    is more sensitive to initialization and requires learning-rate warmup;
    pre-norm is what made very deep transformers practical to train.
    """

    def __init__(
        self,
        n_embd,
        n_head,
        dropout_rate=0.1
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

        # ------------------------------------------------
        # Dropout applied to the ATTENTION SUB-LAYER'S OUTPUT, before it's
        # added back into the residual stream. Same inverted-dropout math
        # as in FeedForward: zero a random subset of the attention output's
        # channels each step, rescale survivors, so the network can't rely
        # on any single attention "route" always being present.
        # ------------------------------------------------
        self.attn_dropout = tf.keras.layers.Dropout(dropout_rate)

        self.ln2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-5
        )

        self.feed_forward = FeedForward(
            n_embd,
            dropout_rate
        )

    def call(self, x, training=False):

        # ----------------------------------------------------
        # Attention sub-layer
        #
        # x -> LayerNorm -> Attention -> Dropout -> Residual add
        # ----------------------------------------------------

        attn_out = self.attention(
            self.ln1(x)
        )

        attn_out = self.attn_dropout(
            attn_out,
            training=training
        )

        x = x + attn_out

        # ----------------------------------------------------
        # Feed-forward sub-layer
        #
        # x -> LayerNorm -> FFN (dropout inside) -> Residual add
        # ----------------------------------------------------

        x = (
            x
            + self.feed_forward(
                self.ln2(x),
                training=training
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
        n_layer,
        dropout_rate=0.1
    ):

        super().__init__()

        self.block_size = block_size

        # ----------------------------------------------------
        # Token embedding
        #
        # A lookup table: row i is a learned n_embd-dimensional vector
        # representing "what token i means". The model never sees the raw
        # integer id directly past this point — everything downstream
        # operates on this dense vector.
        # ----------------------------------------------------

        self.token_embedding = tf.keras.layers.Embedding(
            vocab_size,
            n_embd
        )

        # ----------------------------------------------------
        # Position embedding
        #
        # Attention itself is permutation-invariant (it treats the input as
        # a SET of tokens, not a SEQUENCE — nothing about Q.K@V cares about
        # order). Positional embeddings are how we inject "where in the
        # sequence am I" back in, by adding a learned vector per position
        # index to each token's embedding.
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
                n_head,
                dropout_rate
            )

            for _ in range(n_layer)
        ]

        # ----------------------------------------------------
        # Final normalization
        #
        # One last LayerNorm before the output head, so the final
        # representation fed into lm_head has stable, consistent scale
        # regardless of how it drifted across n_layer blocks.
        # ----------------------------------------------------

        self.final_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-5
        )

        # ----------------------------------------------------
        # Language model head
        #
        # Projects each position's n_embd-dim vector to vocab_size raw
        # scores ("logits") — one score per possible next character.
        # These are NOT probabilities yet; softmax (inside the loss
        # function or during generation) turns them into a distribution.
        # ----------------------------------------------------

        self.lm_head = tf.keras.layers.Dense(
            vocab_size
        )

    def call(self, idx, training=False):

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
        #
        # Simple element-wise addition fuses "identity" and "location" into
        # one vector. Broadcasting handles the fact that token_embeddings
        # is (B,T,C) while position_embeddings is (T,C) — the same position
        # vector is added to every sequence in the batch.
        # ----------------------------------------------------

        x = (
            token_embeddings
            + position_embeddings
        )

        # ----------------------------------------------------
        # Transformer blocks
        #
        # `training` is threaded through every block so Dropout layers
        # know whether to actually drop units (training=True) or act as
        # the identity function (training=False, e.g. during evaluation
        # or text generation).
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x, training=training)

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
    n_layer=N_LAYER,
    dropout_rate=DROPOUT_RATE
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

_ = model(dummy_input, training=False)


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

# SparseCategoricalCrossentropy(from_logits=True) internally does:
#   1. softmax(logits)  -> turns raw scores into a probability distribution
#   2. -log(prob assigned to the TRUE next token)
# Minimizing this pushes the model to assign higher and higher probability
# to whatever character actually came next in the real text. Using
# from_logits=True (rather than pre-computing softmax yourself) is both
# more numerically stable and slightly faster, since TF fuses the two ops.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True
)


# ============================================================
# 16. OPTIMIZER
# ============================================================

# Adam maintains, per parameter: a running average of the gradient (like
# momentum -> smooths out noisy gradient estimates from mini-batches) and
# a running average of the SQUARED gradient (-> gives each parameter its
# own adaptive step size: parameters with consistently large gradients get
# smaller effective steps, and vice versa). This combination is why Adam
# tends to converge faster and more reliably than plain SGD, especially
# for transformers.
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
    #
    # GradientTape RECORDS every differentiable operation performed
    # inside this block (every matmul, add, softmax, etc. across all
    # n_layer blocks). It builds a computation graph on the fly.
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        logits = model(
            xb,
            training=True   # dropout IS active here
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
    #
    # tape.gradient walks the recorded graph BACKWARD from `loss`,
    # applying the chain rule automatically at every recorded op, to
    # compute d(loss)/d(every trainable variable) in one pass. This is
    # exactly what you'd get if you derived and coded every layer's
    # backward formula by hand (as in a from-scratch NumPy version) —
    # here the framework does it for you.
    # --------------------------------------------------------

    gradients = tape.gradient(
        loss,
        model.trainable_variables
    )

    # --------------------------------------------------------
    # Update weights
    #
    # Each variable moves a small step in the direction that REDUCES the
    # loss the fastest (negative gradient direction), with Adam's adaptive
    # per-parameter scaling applied to that step.
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

    # We don't need gradients while evaluating, and critically we pass
    # training=False so Dropout is OFF — we want to measure the model's
    # true, full-capacity performance, not a noisy dropped-out version.

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
    """
    Autoregressive generation: predict ONE next token, append it to the
    sequence, then feed the (now longer) sequence back in and repeat.
    This loop is identical in spirit to how any GPT-style model generates
    text — always one token at a time, always conditioned on everything
    generated so far (up to the context window limit).
    """

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
        #
        # The model has a position embedding table of size BLOCK_SIZE —
        # it mathematically CANNOT be given more than BLOCK_SIZE tokens
        # of context, so we crop.
        # ----------------------------------------------------

        idx_cond = idx[
            :,
            -BLOCK_SIZE:
        ]

        # ----------------------------------------------------
        # Model prediction
        #
        # training=False -> dropout off, deterministic use of the full
        # trained network (no random zeroing during generation).
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
        #
        # logits has shape (B,T,vocab_size) — a next-token prediction for
        # EVERY position, but we only care about predicting what comes
        # after the very last token we currently have.
        # ----------------------------------------------------

        logits_last = logits[
            :,
            -1,
            :
        ]

        # ----------------------------------------------------
        # Temperature
        #
        # Dividing logits by T before softmax: softmax(z/T).
        # T < 1 exaggerates differences between logits -> sharper,
        #        more confident, more repetitive/deterministic distribution.
        # T > 1 shrinks differences between logits -> flatter, more
        #        uniform -> more randomness/novelty, more mistakes.
        # T = 1 uses the model's logits exactly as learned.
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
        #
        # Sampling (rather than always taking argmax) draws from the full
        # learned distribution, so the same prompt can produce different
        # continuations — this is what gives generated text variety.
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