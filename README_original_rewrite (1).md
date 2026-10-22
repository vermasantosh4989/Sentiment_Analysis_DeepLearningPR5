<div align="center">

# 🎥 Movie Review Sentiment Classifier

### Deep Learning Practical · Text Classification with Recurrent Networks

**From raw review text to a tested sentiment-prediction pipeline**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![Task](https://img.shields.io/badge/Practical-Tasks%201.1--7.7-purple)
![Status](https://img.shields.io/badge/Project-Experimentation-informational)

</div>

---

## Contents

- [Project Summary](#project-summary)
- [Why This Problem Matters](#why-this-problem-matters)
- [Dataset and Labels](#dataset-and-labels)
- [Pipeline Map](#pipeline-map)
- [Task 1 — Preparing Review Text](#task-1--preparing-review-text)
  - [1.1 Dataset loading](#11-dataset-loading)
  - [1.2 Data inspection](#12-data-inspection)
  - [1.3 Duplicate handling](#13-duplicate-handling)
  - [1.4 Text normalization](#14-text-normalization)
  - [1.5 Review-length exploration](#15-review-length-exploration)
  - [1.6 Target encoding and splitting](#16-target-encoding-and-splitting)
  - [1.7 Vocabulary and tokenization](#17-vocabulary-and-tokenization)
  - [1.8 Sequence padding](#18-sequence-padding)
  - [1.9 Prepared-data checks](#19-prepared-data-checks)
- [Task 2 — Baseline and RNN Foundations](#task-2--baseline-and-rnn-foundations)
- [Task 3 — SimpleRNN Experiments](#task-3--simplernn-experiments)
- [Task 4 — Learning Through Time and Gradient Behaviour](#task-4--learning-through-time-and-gradient-behaviour)
- [Task 5 — LSTM Experiments](#task-5--lstm-experiments)
- [Task 6 — GRU Experiments](#task-6--gru-experiments)
- [Task 7 — Evaluation and Practical Inference](#task-7--evaluation-and-practical-inference)
- [Results at a Glance](#results-at-a-glance)
- [Charts and Evidence](#charts-and-evidence)
- [Repository Layout](#repository-layout)
- [Environment Setup](#environment-setup)
- [Re-running the Notebook](#re-running-the-notebook)
- [Observations](#observations)
- [Known Limitations](#known-limitations)
- [Potential Next Steps](#potential-next-steps)
- [Wrap-up](#wrap-up)

---

## Project Summary

This practical develops a binary sentiment classifier for movie reviews. The input is a written review; the output is a predicted sentiment class, **Positive** or **Negative**.

The work is arranged as a learning path rather than only a model leaderboard. It begins with dataset inspection and text preparation, establishes a non-recurrent neural baseline, then explores SimpleRNN, LSTM, and GRU networks. The later tasks inspect training behaviour, compare model quality, review mistakes, and test predictions on new sentences.

### Project snapshot

| Item | Project detail |
|---|---|
| Learning area | Natural language processing and deep learning |
| Task type | Binary text classification |
| Dataset | IMDB movie-review dataset |
| Target labels | `positive`, `negative` |
| Vocabulary cap | 10,000 tokens |
| Sequence length used in the recorded run | 600 |
| Cleaned examples in the recorded run | 49,582 |
| Training examples in the recorded run | 39,665 |
| Test examples in the recorded run | 9,917 |
| Compared architectures | ANN, SimpleRNN, LSTM, Stacked LSTM, GRU |
| Selected model in the recorded comparison | GRU |

> **Reproducibility note:** Metrics shown in this README describe the recorded notebook run. Training time and final metrics can change with software versions, hardware, random initialization, and execution settings.

## Why This Problem Matters

Movie platforms can collect thousands of written opinions. A sentiment classifier can help organize this feedback by estimating whether a review expresses a positive or negative opinion.

This notebook is an educational experiment, not a production moderation system. A predicted label is a model estimate; it should not be treated as a perfect understanding of the review or the reviewer's intent.

### Learning goals

- Inspect and clean text data before modeling.
- Reduce evaluation leakage by splitting data before fitting the tokenizer.
- Convert words into integer sequences of a consistent length.
- Build an ANN baseline and recurrent text classifiers.
- Explain how a recurrent hidden state carries information between time steps.
- Observe the training difficulties associated with gradients through long sequences.
- Compare models using more than accuracy alone.
- Examine classification mistakes and run a small set of custom text tests.

## Dataset and Labels

The notebook uses the IMDB Dataset of 50K Movie Reviews.

**Dataset source:** [IMDB Dataset of 50K Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

| Field | Description |
|---|---|
| `review` | Original text of a movie review |
| `sentiment` | Review sentiment, represented as positive or negative |

The raw dataset is commonly distributed with 50,000 records. In the recorded notebook run, duplicate removal leaves **49,582 records**. The notebook then prepares the text and creates training and test partitions.

### Label representation

| Original value | Numeric value |
|---|---:|
| `negative` | 0 |
| `positive` | 1 |

This encoding matches the binary output layer, which uses a sigmoid activation to produce a positive-class probability.

## Pipeline Map

```text
IMDB CSV
   |
   v
Inspect columns, labels, nulls, and examples
   |
   v
Remove duplicate reviews
   |
   v
Normalize review text
   |
   v
Explore review lengths
   |
   v
Create train/test partitions
   |
   v
Fit tokenizer on training text only
   |
   v
Convert words to token IDs
   |
   v
Pad or truncate to a fixed sequence length
   |
   +-------------------+-------------------+
   |                   |                   |
   v                   v                   v
 ANN baseline       SimpleRNN         Gated networks
                                           |
                                  +--------+--------+
                                  |                 |
                                  v                 v
                                 LSTM              GRU
                                  |
                             Stacked LSTM
   |
   v
Evaluate metrics, curves, errors, and custom predictions
```

---

# Task 1 — Preparing Review Text

## Goal

Turn raw text into model-ready sequences while keeping the evaluation process fair. This stage is important because poor text preparation or data leakage can make later scores misleading.

### 1.1 Dataset loading

The CSV is loaded into a Pandas DataFrame. Keeping the data in a DataFrame makes it easy to inspect column names, dimensions, and example records.

Typical checks in this part include:

```python
df.shape
df.head()
df.columns
```

**What to verify:** the file path is correct and the dataset contains the review text and sentiment columns expected by the notebook.

### 1.2 Data inspection

The notebook inspects the target distribution, missing values, and sample reviews before changing the text.

Useful checks include:

```python
df.info()
df.isnull().sum()
df["sentiment"].value_counts()
```

The raw review examples help identify markup and formatting that should be cleaned. Label counts help determine whether the two sentiment classes are represented evenly.

**Evidence to retain:** dataset shape, target counts, null counts, and a small sample of raw reviews.

### 1.3 Duplicate handling

Exact duplicate records are removed before the train/test split. This order matters: a duplicate review appearing in both partitions could make the test score overly optimistic.

After duplicate removal, the recorded dataset contains **49,582 reviews**.

### 1.4 Text normalization

A text-cleaning step is applied consistently to review content. In the notebook, the cleaning logic:

- converts characters to lowercase;
- removes HTML tags;
- filters unwanted characters;
- keeps apostrophes;
- collapses repeated whitespace.

Apostrophes are kept because common contractions can convey negation:

```text
don't
isn't
wasn't
can't
```

Stop words are not removed as a blanket rule. Short words such as `not` can reverse the meaning of a sentence, and sequence models need access to such context.

### 1.5 Review-length exploration

Review lengths vary substantially. Very short sequences can lose context, while excessively long sequences increase computation and padding requirements.

The notebook examines the distribution with a histogram and boxplot. The recorded configuration uses `MAXLEN = 600`.

<div align="center">

![Review length distribution](plot/length_distribution.png)

<sub>Figure 1 — Distribution of review lengths. This figure should be generated by the notebook and saved at the path shown.</sub>

</div>

<div align="center">

![Distribution of IMDB reviews](plot/Distribution_imbd_review.png)

<sub>This screenshot shows how review lengths are spread across the dataset.</sub>

<sub>Use it to explain why sequence padding and truncation are needed.</sub>

</div>


The sequence length should be selected using the actual distribution from the dataset. The recorded value of 600 is a project configuration, not a universal best value for every text dataset.

### 1.6 Target encoding and splitting

The sentiment values are encoded as binary labels. The notebook then creates training and test partitions before fitting the tokenizer.

Recorded partition sizes:

| Partition | Records |
|---|---:|
| Training | 39,665 |
| Test | 9,917 |
| Total after duplicate removal | 49,582 |

The split is designed to preserve an unseen test set for final evaluation. The tokenizer must not learn its vocabulary from the test reviews.

### 1.7 Vocabulary and tokenization

The tokenizer converts each review into a sequence of integer token IDs. The notebook caps the vocabulary at **10,000 tokens** and uses an out-of-vocabulary marker, `<OOV>`, for words not covered by the fitted vocabulary.

Conceptually:

```text
"this movie was excellent"
            |
            v
[token_id, token_id, token_id, token_id]
```

The tokenizer is fitted on training text only. That way, the held-out test set does not influence vocabulary construction.

### 1.8 Sequence padding

Neural networks in the notebook receive sequences with a consistent length. Short sequences are padded; long sequences are truncated.

The recorded configuration uses:

```python
padding="pre"
truncating="pre"
```

Pre-padding inserts zeros at the beginning of shorter sequences. Pre-truncation removes tokens from the beginning of sequences that exceed the maximum length, retaining the later part of the review.

### 1.9 Prepared-data checks

Before training, inspect the resulting array dimensions and confirm that the input and target rows line up.

Checklist:

- [ ] Train and test data contain the expected number of records.
- [ ] Token IDs are integer values.
- [ ] Every padded sequence has the configured length.
- [ ] Labels are encoded consistently.
- [ ] The tokenizer was fitted only on training text.
- [ ] The test partition was not used to fit model parameters.

**Task 1 outcome:** review text is represented as fixed-length token sequences ready for neural-network experiments.

---

# Task 2 — Baseline and RNN Foundations

## Goal

Create a reference classifier and build intuition for the recurrent computation used in later models.

### 2.1 ANN baseline

The baseline architecture uses an embedding representation, flattens the sequence representation, and passes it through dense layers.

```text
Token sequence
      |
   Embedding
      |
    Flatten
      |
 Dense(64, ReLU)
      |
 Dense(1, Sigmoid)
      |
 Sentiment probability
```

The ANN provides a useful comparison point. Because the sequence is flattened, the model does not apply the same recurrent transition at every time step.

### 2.2 Recorded ANN performance

| Metric | Recorded result |
|---|---:|
| Test accuracy | 86.62% |
| Precision | 86.63% |
| Recall | 86.72% |
| F1-score | 86.68% |
| ROC-AUC | 0.9379 |
| Parameters | 1,548,929 |
| Time per epoch | 12.60 s |

These results are included as a baseline for comparison with the recurrent models.

### 2.3 Recurrent-network intuition

A recurrent neural network processes a sequence one step at a time. The current hidden state is calculated from the current input and the previous hidden state.

\[
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)
\]

Here, \(x_t\) is the current input, \(h_{t-1}\) is the previous hidden state, and \(h_t\) is the new hidden state. The recurrent weights are reused at each time step.

### 2.4 Manual NumPy forward pass

A small RNN is implemented manually to make the calculation easier to follow:

| Setting | Value |
|---|---:|
| Input dimension | 2 |
| Hidden-state dimension | 3 |
| Sequence steps | 4 |

At every step, the current input and previous hidden state are combined, passed through a non-linear activation, and used as the next hidden state.

<div align="center">

![Manual RNN hidden state heatmap](plot/rnn_forward_heatmap.png)

<sub>Figure 2 — Hidden-state values across time steps for the small NumPy RNN.</sub>

</div>

### 2.5 Compare manual and Keras calculations

The notebook transfers compatible weights from the NumPy implementation into a Keras `SimpleRNN` and compares the outputs for the same input sequence. This is a practical way to check the manual equations against a framework implementation.

### 2.6 ANN and RNN comparison

| Characteristic | ANN baseline | SimpleRNN |
|---|---|---|
| Sequence handling | Flattened representation | Step-by-step recurrence |
| Shared recurrent transition | No | Yes |
| Carries hidden state across tokens | No recurrent state | Yes |
| Main role in this practical | Baseline | Sequential-model experiment |

**Task 2 outcome:** the baseline score is established, and the hidden-state update that powers recurrent models is demonstrated.

---

# Task 3 — SimpleRNN Experiments

## Goal

Train a recurrent model on the padded reviews and evaluate both predictive quality and training behaviour.

### 3.1 Model layout

The SimpleRNN classifier uses an embedding layer, a recurrent layer, and a sigmoid output.

```text
Token IDs
   |
Embedding (vocabulary 10,000; dimension 32)
   |
SimpleRNN (64 units)
   |
Dense (1 unit, sigmoid)
   |
Positive-class probability
```

### 3.2 Training approach

The notebook compiles the classifier for binary classification and trains it using a validation split. Early stopping is used in the model-training workflow to avoid continuing unnecessarily when validation loss stops improving.

<div align="center">

![ANN training and validation accuracy](plot/ANN_Training&Validation_accuracy.png)

<sub>This plot compares ANN accuracy on training data with validation accuracy.</sub>

<sub>A widening gap between the two curves can be a sign of overfitting.</sub>

</div>


<div align="center">

![ANN training and validation loss](plot/ANN_Training&Validation_loss.png)

<sub>This plot tracks the ANN loss during training and validation.</sub>

<sub>Validation loss helps assess whether learning is generalizing to unseen examples.</sub>

</div>


Training curves should be read together: training performance can continue improving even when validation performance stops improving, which may indicate overfitting.

<div align="center">

![SimpleRNN training curves](plot/training_curves_simplernn.png)

<sub>Figure 3 — SimpleRNN training and validation behaviour.</sub>

</div>

<div align="center">

![SimpleRNN loss and accuracy](plot/SimpleRNN_loss_&_Accuracy.png)

<sub>This figure summarizes SimpleRNN training and validation metrics across epochs.</sub>

<sub>Compare both metrics together to judge learning progress and possible overfitting.</sub>

</div>


### 3.3 Recorded SimpleRNN results

| Metric | Recorded result |
|---|---:|
| Test accuracy | 79.01% |
| Precision | 88.69% |
| Recall | 66.67% |
| F1-score | 76.12% |
| ROC-AUC | 0.8787 |
| Parameters | 326,273 |
| Epochs recorded | 4 |

The relatively low recall means that the model misses a noticeable share of positive reviews under the selected classification threshold. This is why accuracy should not be the only metric used for model selection.

### 3.4 Confusion matrix

<div align="center">

![SimpleRNN confusion matrix](plot/confusion_matrix_simplernn.png)

<sub>Figure 4 — Confusion matrix for the SimpleRNN test predictions.</sub>

</div>

<div align="center">

![SimpleRNN confusion matrix screenshot](plot/SimpleRNN_confusion_matrix.png)

<sub>This screenshot displays correct and incorrect SimpleRNN predictions by class.</sub>

<sub>It helps distinguish false positives from false negatives.</sub>

</div>


A confusion matrix separates correct positive and negative predictions from false positives and false negatives. It makes the types of mistakes easier to interpret than a single accuracy value.

### 3.5 Hidden-state and sequence behaviour

The recurrent state is updated for each token. This provides a mechanism for carrying information forward, but learning dependencies across long sequences can be difficult with a basic recurrent cell.

<div align="center">

![RNN hidden states](plot/RNN_hidden_states.png)

<sub>This screenshot visualizes hidden-state values produced as the RNN processes a sequence.</sub>

<sub>It gives an intuitive view of how the representation changes across time steps.</sub>

</div>


**Task 3 outcome:** SimpleRNN serves as the first sequence-aware model and establishes a reference for the gated architectures.

---

# Task 4 — Learning Through Time and Gradient Behaviour

## Goal

Understand how the training signal moves through a recurrent sequence and why long dependencies can be difficult to learn.

### 4.1 Backpropagation Through Time

Backpropagation Through Time (BPTT) applies the chain rule through the sequence of recurrent state updates. During training, errors at the output contribute gradients to parameters used across multiple time steps.

As a sequence becomes longer, the gradient calculation passes through more recurrent transitions. Repeated multiplication by derivatives can cause gradients to shrink or grow.

### 4.2 Vanishing gradients

When gradients become very small, early time steps receive a weak learning signal. The model may struggle to learn relationships between words that are far apart in a review.

### 4.3 Exploding gradients

When gradients become excessively large, updates can become unstable. The notebook records gradient behaviour to help illustrate this optimization problem.

<div align="center">

![SimpleRNN gradient norm](plot/gradient_norm_simplernn.png)

<sub>Figure 5 — Gradient-norm behaviour during the SimpleRNN experiment.</sub>

</div>

<div align="center">

![SimpleRNN gradient norm screenshot](plot/SimpleRNN_gradient_norm.png)

<sub>The plot tracks gradient magnitude during the recurrent training experiment.</sub>

<sub>Very small or very large values can indicate gradient-flow difficulties.</sub>

</div>


### 4.4 Gradient clipping

Gradient clipping limits the magnitude of gradients before the optimizer applies an update. It is a stabilizing technique; it does not guarantee better test performance in every run.

<div align="center">

![Gradient clipping comparison](plot/gradient_clipping_comparison.png)

<sub>Figure 6 — Training behaviour with and without gradient clipping.</sub>

</div>

<div align="center">

![SimpleRNN with and without gradient clipping](plot/SimpleRNN_loss_with_vs_without_gradient.png)

<sub>This screenshot compares training behaviour when gradient clipping is enabled or disabled.</sub>

<sub>It illustrates how clipping may help stabilize optimization.</sub>

</div>


### 4.5 Sequence-length sensitivity

The practical compares model behaviour at different maximum sequence lengths. This helps show the trade-off between retaining more context and increasing computational work.

<div align="center">

![RNN accuracy versus sequence length](plot/accuracy_vs_sequence_length_rnn.png)

<sub>Figure 7 — Accuracy under different sequence-length settings.</sub>

</div>

<div align="center">

![Sequence-length robustness](plot/sequence_length_robustness.png)

<sub>This plot compares sentiment accuracy across review-length groups or sequence settings.</sub>

<sub>It helps show whether longer reviews noticeably change model performance.</sub>

</div>


Recorded sequence-length results:

| Model | Short reviews | Medium reviews | Long reviews |
|---|---:|---:|---:|
| SimpleRNN | 78.83% | 80.20% | 76.67% |
| GRU | 90.17% | 89.98% | 89.05% |

The exact interpretation depends on the notebook's definitions of short, medium, and long review groups. These results indicate that the recorded GRU run was more consistent across those groups than the SimpleRNN run.

### 4.6 GradientTape experiment

The notebook uses TensorFlow's gradient tools to inspect gradients through recurrent computations. The goal is to connect the mathematical explanation of BPTT to quantities measured during model execution.

**Task 4 outcome:** gradient diagnostics explain why gated recurrent cells are worth exploring, while clipping offers one practical tool for training stability.

---

# Task 5 — LSTM Experiments

## Goal

Evaluate a Long Short-Term Memory network, which uses gates and a cell state to regulate information flow through a sequence.

### 5.1 Why use LSTM?

A basic RNN repeatedly updates its hidden state. LSTM adds a cell state and gates that control which information is forgotten, added, and exposed. This design can make learning longer dependencies easier, although it increases computation.

### 5.2 Model layout

```text
Token IDs
   |
Embedding (10,000 words; 32 dimensions)
   |
LSTM (64 units)
   |
Dense (1 unit, sigmoid)
   |
Sentiment prediction
```

### 5.3 Training curves

<div align="center">

![LSTM training curves](plot/training_curves_lstm.png)

<sub>Figure 8 — LSTM training and validation curves.</sub>

</div>

<div align="center">

![LSTM loss and accuracy](plot/LSTM_loss_vs_accuracy.png)

<sub>This figure shows how LSTM accuracy and loss change over training epochs.</sub>

<sub>Training and validation trends help evaluate generalization.</sub>

</div>


### 5.4 Confusion matrix

<div align="center">

![LSTM confusion matrix](plot/confusion_matrix_lstm.png)

<sub>Figure 9 — LSTM prediction outcomes on the test partition.</sub>

</div>

<div align="center">

![LSTM confusion matrix screenshot](plot/LSTM_confusion_matrix.png)

<sub>This screenshot summarizes the LSTM model’s test predictions by actual and predicted class.</sub>

<sub>Use it to discuss the model’s false-positive and false-negative errors.</sub>

</div>


### 5.5 Gradient comparison with SimpleRNN

<div align="center">

![RNN and LSTM gradient comparison](plot/gradient_norm_rnn_vs_lstm.png)

<sub>Figure 10 — Gradient-norm comparison between recurrent architectures.</sub>

</div>

<div align="center">

![SimpleRNN versus LSTM gradient comparison](plot/Gradint_simplernn_vs_lstm.png)

<sub>This plot compares gradient behaviour for the basic RNN and LSTM.</sub>

<sub>It provides evidence for discussing how gated memory affects gradient flow.</sub>

</div>


### 5.6 Recorded LSTM results

| Metric | Recorded result |
|---|---:|
| Test accuracy | 88.68% |
| Precision | 87.13% |
| Recall | 90.86% |
| F1-score | 88.95% |
| ROC-AUC | 0.9497 |
| Parameters | 344,897 |
| Epochs recorded | 7 |
| Time per epoch | 179.99 s |

LSTM records the highest recall among the final model rows shown here. This can be important when missing a positive review is more costly than incorrectly marking a negative review as positive.

### 5.7 Stacked LSTM extension

A stacked recurrent model adds another LSTM layer. Additional depth can increase model capacity, but it also increases training cost and does not automatically improve generalization.

<div align="center">

![Stacked LSTM confusion matrix](plot/Stacked_LSTM_confusion_matrix.png)

<sub>This screenshot summarizes the stacked LSTM predictions by actual and predicted sentiment.</sub>

<sub>Compare the error pattern with the single-layer LSTM and GRU models.</sub>

</div>

Recorded stacked-LSTM results:

| Metric | Recorded result |
|---|---:|
| Test accuracy | 88.31% |
| Precision | 88.58% |
| Recall | 88.07% |
| F1-score | 88.32% |
| ROC-AUC | 0.9499 |
| Parameters | 357,281 |
| Epochs recorded | 6 |
| Time per epoch | 1,358.36 s |

The recorded run does not show a performance improvement over the single-layer LSTM or GRU. Runtime can vary substantially with hardware and implementation details.

<div align="center">

![Stacked LSTM loss and accuracy](plot/Stacked_LSTM_loss_vs_accuracy.png)

<sub>This plot tracks the stacked LSTM’s training and validation performance.</sub>

<sub>It can help assess whether the additional recurrent layer improves learning.</sub>

</div>


**Task 5 outcome:** LSTM improves on the SimpleRNN baseline in the recorded experiment and demonstrates the value of gated memory.

---

# Task 6 — GRU Experiments

## Goal

Train a Gated Recurrent Unit network and compare its balance of predictive quality, parameter count, and training time with LSTM.

### 6.1 GRU idea

GRU uses gates to regulate how information from earlier steps contributes to the current representation. It has a different gate structure from LSTM and does not maintain a separate cell state in the same way.

### 6.2 Model layout

```text
Token IDs
   |
Embedding
   |
GRU (64 units)
   |
Dense (1 unit, sigmoid)
   |
Positive-class probability
```

### 6.3 Training behaviour

<div align="center">

![GRU training curves](plot/training_curves_gru.png)

<sub>Figure 11 — GRU training and validation performance.</sub>

</div>

<div align="center">

![GRU loss and accuracy](plot/GRU_loss_vs_accuracy.png)

<sub>This figure compares the GRU’s loss and accuracy over training epochs.</sub>

<sub>Review training and validation curves together to check model fit.</sub>

</div>


### 6.4 Confusion matrix

<div align="center">

![GRU confusion matrix](plot/confusion_matrix_gru.png)

<sub>Figure 12 — GRU classification outcomes on held-out reviews.</sub>

</div>

<div align="center">

![GRU confusion matrix screenshot](plot/GRU_confusion_matrix.png)

<sub>This screenshot shows the GRU’s correct and incorrect predictions on the test set.</sub>

<sub>It complements accuracy, precision, recall, and F1-score.</sub>

</div>


### 6.5 Gradient comparison across recurrent cells

<div align="center">

![Gradient norms for recurrent cells](plot/gradient_norms_all_cells.png)

<sub>Figure 13 — Gradient diagnostics across SimpleRNN, LSTM, and GRU experiments.</sub>

</div>

<div align="center">

![Gradient flow across recurrent cells](plot/Gradient_flow_simpleRNN_vs_LSTM_VS_GRU.png)

<sub>This comparison shows gradient behaviour for SimpleRNN, LSTM, and GRU.</sub>

<sub>It supports a side-by-side discussion of how the recurrent architectures learn.</sub>

</div>


### 6.6 Hidden-unit sensitivity

Changing the number of hidden units affects both model capacity and parameter count. The notebook records how the results vary with this setting.

<div align="center">

![Accuracy versus hidden units](plot/accuracy_vs_hidden_units.png)

<sub>Figure 14 — Accuracy across hidden-unit configurations.</sub>

</div>

<div align="center">

![Accuracy versus hidden units screenshot](plot/accuracy_vs_hidden_units.png)

<sub>This plot compares model accuracy for different hidden-unit counts.</sub>

<sub>It helps assess whether a larger recurrent layer provides a meaningful benefit.</sub>

</div>


<div align="center">

![Parameters versus hidden units](plot/parameters_vs_hidden_units.png)

<sub>Figure 15 — Model parameter growth as hidden units increase.</sub>

</div>

### 6.7 Recorded GRU results

| Metric | Recorded result |
|---|---:|
| Test accuracy | 89.73% |
| Precision | 89.69% |
| Recall | 89.87% |
| F1-score | 89.78% |
| ROC-AUC | 0.9595 |
| Parameters | 338,881 |
| Epochs recorded | 6 |
| Time per epoch | 174.53 s |

In the recorded comparison, GRU achieves the strongest overall combination of accuracy, F1-score, ROC-AUC, parameter count, and measured epoch time among the gated candidates.

**Task 6 outcome:** GRU becomes the preferred model for the final comparison in this particular run.

---

# Task 7 — Evaluation and Practical Inference

## Goal

Bring the model experiments together, inspect their errors, test new sentences, and state a recommendation based on the recorded evidence.

### 7.1 Compare the model results

The final table uses several metrics because each describes a different aspect of model performance.

- **Accuracy:** proportion of all test examples classified correctly.
- **Precision:** among predicted positives, the proportion that are truly positive.
- **Recall:** among actual positives, the proportion detected by the model.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** ranking performance across classification thresholds.
- **Parameter count:** rough indication of model size, not a direct measure of inference speed.

### 7.2 Combined model comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| ANN | 86.62% | 86.63% | 86.72% | 86.68% | 0.9379 | 1,548,929 |
| SimpleRNN | 79.01% | 88.69% | 66.67% | 76.12% | 0.8787 | 326,273 |
| LSTM | 88.68% | 87.13% | 90.86% | 88.95% | 0.9497 | 344,897 |
| Stacked LSTM | 88.31% | 88.58% | 88.07% | 88.32% | 0.9499 | 357,281 |
| GRU | **89.73%** | **89.69%** | 89.87% | **89.78%** | **0.9595** | **338,881** |

<div align="center">

![Final results table](plot/results_table.png)

<sub>Figure 16 — Final metrics visualized from the notebook comparison.</sub>

</div>

<div align="center">

![LSTM hidden states and gradient loss](plot/LSTM_hidden_states_gradint%20loss.png)

<sub>This screenshot captures the recorded LSTM state or gradient-loss diagnostic.</sub>

<sub>Use it alongside the gradient comparison to discuss recurrent learning behaviour.</sub>

</div>


### 7.3 ROC curve comparison

ROC curves show how the true-positive rate changes against the false-positive rate as the classification threshold varies.

<div align="center">

![ROC curves](plot/roc_curves.png)

<sub>Figure 17 — ROC curve comparison for the evaluated models.</sub>

</div>

<div align="center">

![ROC curve comparison screenshot](plot/roc_curve_comparison.png)

<sub>This figure compares the true-positive and false-positive rates across classification thresholds.</sub>

<sub>A larger ROC-AUC indicates stronger ranking performance across thresholds.</sub>

</div>


The GRU's recorded ROC-AUC of 0.9595 is the highest in the table. ROC-AUC complements threshold-specific measures; it does not replace checking precision and recall at the threshold used by the application.

### 7.4 Review-length robustness

The notebook groups reviews by length to test whether the model performs similarly on shorter and longer text.

<div align="center">

![Accuracy by review length](plot/accuracy_by_length.png)

<sub>Figure 18 — Accuracy grouped by review length.</sub>

</div>

<div align="center">

![Review-length robustness screenshot](plot/sequence_length_robustness.png)

<sub>This screenshot compares accuracy across review-length categories.</sub>

<sub>It helps check whether the model remains reliable on long as well as short reviews.</sub>

</div>


The recorded GRU accuracies are 90.17% for short reviews, 89.98% for medium reviews, and 89.05% for long reviews. The SimpleRNN values are 78.83%, 80.20%, and 76.67%, respectively.

### 7.5 Error inspection

The error-analysis step examines false positives and false negatives rather than treating all incorrect predictions as interchangeable.

In the recorded GRU analysis:

| Error category | Count |
|---|---:|
| False positives | 514 |
| False negatives | 504 |

A false positive is a negative review predicted as positive. A false negative is a positive review predicted as negative. The notebook prints example reviews with their true labels, predicted labels, and predicted positive probabilities.

<div align="center">

![Sentiment review prediction examples](plot/sentiment_review.png)

<sub>This screenshot shows review text alongside predicted sentiment information.</sub>

<sub>It makes the model’s decisions easier to inspect with concrete examples.</sub>

</div>


Recorded example probabilities:

| Example | Positive-class probability |
|---|---:|
| False positive 1 | 0.6195 |
| False positive 2 | 0.7703 |
| False positive 3 | 0.7146 |
| False positive 4 | 0.6687 |
| False negative 1 | 0.3560 |
| False negative 2 | 0.1355 |
| False negative 3 | 0.0753 |
| False negative 4 | 0.4830 |

Potential reasons for mistakes include mixed opinions, sarcasm, negation, sentiment changes within a long review, and ambiguous wording. These are plausible interpretations of text-classification errors, not proof of the cause of every individual error.

### 7.6 Custom text predictions

The notebook defines a reusable prediction function that follows the same preprocessing steps as training:

```text
New sentence
     |
Text cleaning
     |
Training tokenizer
     |
Token sequence
     |
Padding to the selected length
     |
Selected model prediction
     |
Probability and class label
```

Example usage:

```python
predict_sentiment(
    "This movie was absolutely wonderful and I loved it."
)
```

The recorded example output is:

```python
{
    "label": "Positive",
    "positive_probability": 0.9850381016731262
}
```

The six custom language tests recorded in the notebook are summarized below.

| Test case | Predicted label | Positive probability |
|---|---|---:|
| Clearly positive | Positive | 0.9787 |
| Clearly negative | Negative | 0.0053 |
| Negation | Negative | 0.1630 |
| Double negation | Negative | 0.2545 |
| Sarcastic example | Negative | 0.4140 |
| Very short positive text | Positive | 0.9589 |

These are examples from the recorded run, not guarantees about how the model will respond to every sentence of the same type. In particular, double negation and sarcasm are difficult linguistic patterns, and the table illustrates that the model can still misinterpret them.

### 7.7 Final model recommendation

The recorded results support choosing **GRU** as the overall model for this practical.

| Decision factor | Preferred result in the recorded run |
|---|---|
| Highest accuracy | GRU — 89.73% |
| Highest precision | GRU — 89.69% |
| Highest recall | LSTM — 90.86% |
| Highest F1-score | GRU — 89.78% |
| Highest ROC-AUC | GRU — 0.9595 |
| Lowest parameter count among recurrent models listed | SimpleRNN — 326,273 |
| Best balance among gated candidates | GRU |

The recommendation depends on the evaluation priorities. If recall is the dominant requirement, LSTM deserves consideration because it records the highest recall. For the overall combination of metrics in this run, GRU is the preferred candidate.

---

## Results at a Glance

| Architecture | Accuracy | Main observation |
|---|---:|---|
| ANN | 86.62% | Useful baseline, relatively high parameter count |
| SimpleRNN | 79.01% | Lowest accuracy and recall in the comparison |
| LSTM | 88.68% | Strong recall |
| Stacked LSTM | 88.31% | More computation without the best final score |
| GRU | **89.73%** | Best overall balance in this recorded run |

<div align="center">

![Combined recurrent training curves](plot/training_curves_rnn_lstm_gru.png)

<sub>Figure 19 — Combined training curves for the recurrent models.</sub>

</div>


## Additional Screenshot Gallery

The following images are expected inside the `plot/` folder. Each caption briefly explains what the screenshot documents and where it is useful in the report.



<div align="center">

![Accuracy versus hidden units](plot/accuracy_vs_hidden_units.png)

<sub>Compares model accuracy at different hidden-unit counts.</sub>

<sub>Place it in the GRU/model-capacity comparison section.</sub>

</div>


<div align="center">

![ANN training and validation accuracy](plot/ANN_Training&Validation_accuracy.png)

<sub>Shows accuracy trends for training and validation data.</sub>

<sub>Use it to discuss convergence and possible overfitting.</sub>

</div>


<div align="center">

![ANN training and validation loss](plot/ANN_Training&Validation_loss.png)

<sub>Shows loss trends during ANN training.</sub>

<sub>Use validation loss to assess generalization.</sub>

</div>


<div align="center">

![IMDB review distribution](plot/Distribution_imbd_review.png)

<sub>Shows the distribution of review lengths in the dataset.</sub>

<sub>Place it in the data exploration section.</sub>

</div>


<div align="center">

![Gradient flow across recurrent models](plot/Gradient_flow_simpleRNN_vs_LSTM_VS_GRU.png)

<sub>Compares gradient flow in SimpleRNN, LSTM, and GRU.</sub>

<sub>Place it in the gradient-behaviour section.</sub>

</div>


<div align="center">

![SimpleRNN versus LSTM gradients](plot/Gradint_simplernn_vs_lstm.png)

<sub>Compares gradient behaviour for SimpleRNN and LSTM.</sub>

<sub>Place it in the BPTT/gradient discussion.</sub>

</div>


<div align="center">

![GRU confusion matrix](plot/GRU_confusion_matrix.png)

<sub>Summarizes correct and incorrect GRU predictions.</sub>

<sub>Place it beside the GRU evaluation results.</sub>

</div>


<div align="center">

![GRU loss and accuracy](plot/GRU_loss_vs_accuracy.png)

<sub>Shows GRU learning curves across epochs.</sub>

<sub>Place it in the GRU training section.</sub>

</div>


<div align="center">

![LSTM confusion matrix](plot/LSTM_confusion_matrix.png)

<sub>Summarizes LSTM test predictions by class.</sub>

<sub>Place it beside the LSTM evaluation results.</sub>

</div>


<div align="center">

![LSTM hidden states and gradient loss](plot/LSTM_hidden_states_gradint%20loss.png)

<sub>Documents the LSTM state/gradient diagnostic.</sub>

<sub>Place it in the LSTM gradient analysis section.</sub>

</div>


<div align="center">

![LSTM loss and accuracy](plot/LSTM_loss_vs_accuracy.png)

<sub>Shows LSTM training and validation trends.</sub>

<sub>Place it in the LSTM training section.</sub>

</div>


<div align="center">

![RNN hidden states](plot/RNN_hidden_states.png)

<sub>Visualizes hidden-state changes across sequence steps.</sub>

<sub>Place it in the manual RNN explanation.</sub>

</div>


<div align="center">

![ROC curve comparison](plot/roc_curve_comparison.png)

<sub>Compares classification performance across thresholds.</sub>

<sub>Place it in the final model comparison.</sub>

</div>


<div align="center">

![Sentiment review examples](plot/sentiment_review.png)

<sub>Shows example reviews and model predictions.</sub>

<sub>Place it in the inference/error inspection section.</sub>

</div>


<div align="center">

![Sequence-length robustness](plot/sequence_length_robustness.png)

<sub>Compares performance for different review lengths.</sub>

<sub>Place it in the robustness section.</sub>

</div>


<div align="center">

![SimpleRNN confusion matrix](plot/SimpleRNN_confusion_matrix.png)

<sub>Summarizes SimpleRNN test prediction outcomes.</sub>

<sub>Place it beside the SimpleRNN evaluation results.</sub>

</div>


<div align="center">

![SimpleRNN gradient norm](plot/SimpleRNN_gradient_norm.png)

<sub>Shows gradient magnitude during training.</sub>

<sub>Place it in the gradient-flow section.</sub>

</div>


<div align="center">

![SimpleRNN loss and accuracy](plot/SimpleRNN_loss_&_Accuracy.png)

<sub>Shows the SimpleRNN training metrics.</sub>

<sub>Place it in the SimpleRNN training section.</sub>

</div>


<div align="center">

![SimpleRNN gradient clipping comparison](plot/SimpleRNN_loss_with_vs_without_gradient.png)

<sub>Compares optimization with and without gradient clipping.</sub>

<sub>Place it in the gradient-clipping experiment.</sub>

</div>


<div align="center">

![Stacked LSTM confusion matrix](plot/Stacked_LSTM_confusion_matrix.png)

<sub>Summarizes stacked LSTM classification outcomes.</sub>

<sub>Place it in the stacked-LSTM evaluation section.</sub>

</div>


<div align="center">

![Stacked LSTM loss and accuracy](plot/Stacked_LSTM_loss_vs_accuracy.png)

<sub>Shows training and validation metrics for stacked LSTM.</sub>

<sub>Place it in the stacked-LSTM training section.</sub>

</div>

## Charts and Evidence

The notebook can save figures in a `plot/` directory. The Markdown image links below expect those files to exist at the corresponding relative paths.

| File | What it documents |
|---|---|
| `length_distribution.png` | Distribution of review lengths |
| `rnn_forward_heatmap.png` | Hidden-state values in the manual RNN |
| `training_curves_simplernn.png` | SimpleRNN learning curves |
| `confusion_matrix_simplernn.png` | SimpleRNN classification errors |
| `gradient_norm_simplernn.png` | SimpleRNN gradient norms |
| `accuracy_vs_sequence_length_rnn.png` | Performance at different sequence lengths |
| `gradient_clipping_comparison.png` | Gradient-clipping experiment |
| `training_curves_lstm.png` | LSTM learning curves |
| `confusion_matrix_lstm.png` | LSTM classification errors |
| `gradient_norm_rnn_vs_lstm.png` | Gradient comparison for RNN and LSTM |
| `training_curves_gru.png` | GRU learning curves |
| `confusion_matrix_gru.png` | GRU classification errors |
| `gradient_norms_all_cells.png` | Gradient comparison across recurrent cells |
| `accuracy_vs_hidden_units.png` | Accuracy versus hidden-unit count |
| `parameters_vs_hidden_units.png` | Parameter count versus hidden-unit count |
| `results_table.png` | Rendered final metrics table |
| `roc_curves.png` | ROC curves |
| `accuracy_by_length.png` | Accuracy by review-length group |
| `training_curves_rnn_lstm_gru.png` | Combined recurrent training curves |
| `ANN_Training&Validation_accuracy.png` | ANN training and validation accuracy |
| `ANN_Training&Validation_loss.png` | ANN training and validation loss |
| `Distribution_imbd_review.png` | IMDB review length distribution |
| `Gradient_flow_simpleRNN_vs_LSTM_VS_GRU.png` | Gradient flow across recurrent cells |
| `Gradint_simplernn_vs_lstm.png` | SimpleRNN versus LSTM gradients |
| `GRU_confusion_matrix.png` | GRU confusion matrix |
| `GRU_loss_vs_accuracy.png` | GRU loss and accuracy |
| `LSTM_confusion_matrix.png` | LSTM confusion matrix |
| `LSTM_hidden_states_gradint loss.png` | LSTM hidden-state/gradient diagnostic |
| `LSTM_loss_vs_accuracy.png` | LSTM loss and accuracy |
| `RNN_hidden_states.png` | RNN hidden states |
| `roc_curve_comparison.png` | ROC curve comparison |
| `sentiment_review.png` | Review sentiment examples |
| `sequence_length_robustness.png` | Sequence-length robustness |
| `SimpleRNN_confusion_matrix.png` | SimpleRNN confusion matrix |
| `SimpleRNN_gradient_norm.png` | SimpleRNN gradient norm |
| `SimpleRNN_loss_&_Accuracy.png` | SimpleRNN loss and accuracy |
| `SimpleRNN_loss_with_vs_without_gradient.png` | Gradient clipping comparison |
| `Stacked_LSTM_confusion_matrix.png` | Stacked LSTM confusion matrix |
| `Stacked_LSTM_loss_vs_accuracy.png` | Stacked LSTM loss and accuracy |

### Screenshot and plot placement guidance

1. Save each chart using the filename shown in the table.
2. Put all chart files inside a folder named `plots`.
3. Keep capitalization and underscores consistent between filenames and Markdown references.
4. Open the README preview locally before pushing it to GitHub.
5. If a chart was not generated, remove its image link or generate the missing chart in the notebook. Do not leave a broken image path in the final repository.

---


### Task 1 — Text preparation review checklist

These points can be used when checking the Task 1 notebook section before submission:

- [ ] Check the original columns before selecting the fields used by the model.
- [ ] Inspect missing values and target counts before cleaning.
- [ ] Remove exact duplicates before creating the held-out test partition.
- [ ] Apply the same cleaning rules to training, test, and future user input.
- [ ] Fit the tokenizer only on the training text.
- [ ] Check the final padded sequence shape before model construction.

**Submission evidence:** keep the relevant printed output, chart, or result table near this part of the notebook so the README and notebook can be checked together.


### Task 2 — Baseline and recurrence review checklist

These points can be used when checking the Task 2 notebook section before submission:

- [ ] Record the baseline metrics before comparing more complex models.
- [ ] Check the output layer and loss function match binary classification.
- [ ] Follow one recurrent time step at a time in the manual example.
- [ ] Compare compatible NumPy and Keras outputs using the same inputs and weights.

**Submission evidence:** keep the relevant printed output, chart, or result table near this part of the notebook so the README and notebook can be checked together.


### Task 3 — SimpleRNN review checklist

These points can be used when checking the Task 3 notebook section before submission:

- [ ] Check that the embedding output has the expected sequence dimensions.
- [ ] Review training and validation curves together.
- [ ] Read the confusion matrix alongside recall and precision.
- [ ] Keep the trained model and its metrics associated with the same run.

**Submission evidence:** keep the relevant printed output, chart, or result table near this part of the notebook so the README and notebook can be checked together.


### Task 4 — Gradient experiment review checklist

These points can be used when checking the Task 4 notebook section before submission:

- [ ] Explain BPTT before interpreting gradient measurements.
- [ ] Use consistent logging intervals when comparing gradient norms.
- [ ] Describe clipping as a stabilization technique rather than a guaranteed accuracy improvement.
- [ ] Keep sequence-length experiments comparable wherever possible.

**Submission evidence:** keep the relevant printed output, chart, or result table near this part of the notebook so the README and notebook can be checked together.


### Task 5 — LSTM review checklist

These points can be used when checking the Task 5 notebook section before submission:

- [ ] Describe the purpose of gates and the cell state in your own words.
- [ ] Compare the LSTM with SimpleRNN using the same held-out test data.
- [ ] Record epoch count and runtime as environment-specific observations.
- [ ] Do not assume a stacked architecture must outperform a single recurrent layer.

**Submission evidence:** keep the relevant printed output, chart, or result table near this part of the notebook so the README and notebook can be checked together.


### Task 6 — GRU review checklist

These points can be used when checking the Task 6 notebook section before submission:

- [ ] Explain that GRU uses gates to control information updates.
- [ ] Compare accuracy, precision, recall, F1, and ROC-AUC together.
- [ ] Include parameter count and training time when discussing efficiency.
- [ ] Use the same evaluation split when comparing candidate architectures.

**Submission evidence:** keep the relevant printed output, chart, or result table near this part of the notebook so the README and notebook can be checked together.


### Task 7 — Final evaluation review checklist

These points can be used when checking the Task 7 notebook section before submission:

- [ ] Report the chosen threshold used to turn probabilities into labels.
- [ ] Review false positives and false negatives separately.
- [ ] Test custom examples that include positive, negative, negation, and ambiguous wording.
- [ ] State why the final model is recommended and mention any metric where another model wins.
- [ ] Check that every image path in the README points to a real file in the repository.

**Submission evidence:** keep the relevant printed output, chart, or result table near this part of the notebook so the README and notebook can be checked together.


## Repository Layout

One possible project layout is:

```text
movie-review-sentiment/
├── README.md
├── requirements.txt
├── IMDB_Sentiment_Analysis.ipynb
├── IMDB_Sentiment_Analysis.html
├── model_comparison_results.csv
├── data/
│   └── IMDB Dataset.csv
└── plot/
    ├── length_distribution.png
    ├── rnn_forward_heatmap.png
    ├── training_curves_simplernn.png
    ├── confusion_matrix_simplernn.png
    ├── gradient_norm_simplernn.png
    ├── accuracy_vs_sequence_length_rnn.png
    ├── gradient_clipping_comparison.png
    ├── training_curves_lstm.png
    ├── confusion_matrix_lstm.png
    ├── gradient_norm_rnn_vs_lstm.png
    ├── training_curves_gru.png
    ├── confusion_matrix_gru.png
    ├── gradient_norms_all_cells.png
    ├── accuracy_vs_hidden_units.png
    ├── parameters_vs_hidden_units.png
    ├── results_table.png
    ├── roc_curves.png
    ├── accuracy_by_length.png
    ├── training_curves_rnn_lstm_gru.png
    ├── ANN_Training&Validation_accuracy.png
    ├── ANN_Training&Validation_loss.png
    ├── Distribution_imbd_review.png
    ├── Gradient_flow_simpleRNN_vs_LSTM_VS_GRU.png
    ├── Gradint_simplernn_vs_lstm.png
    ├── GRU_confusion_matrix.png
    ├── GRU_loss_vs_accuracy.png
    ├── LSTM_confusion_matrix.png
    ├── LSTM_hidden_states_gradint loss.png
    ├── LSTM_loss_vs_accuracy.png
    ├── RNN_hidden_states.png
    ├── roc_curve_comparison.png
    ├── sentiment_review.png
    ├── sequence_length_robustness.png
    ├── SimpleRNN_confusion_matrix.png
    ├── SimpleRNN_gradient_norm.png
    ├── SimpleRNN_loss_&_Accuracy.png
    ├── SimpleRNN_loss_with_vs_without_gradient.png
    ├── Stacked_LSTM_confusion_matrix.png
    └── Stacked_LSTM_loss_vs_accuracy.png
```

The folder names are a suggested organization. If the notebook currently expects the CSV in the project root, either keep it there or update the notebook's data path before moving it into `data/`.

## Environment Setup

Python package versions can affect compatibility, especially for TensorFlow. Use a Python version supported by the TensorFlow release installed on your computer.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate it on Windows

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open the project notebook and run its cells in order.

## Example `requirements.txt`

The following is a starting list of direct dependencies used by the project. For a reproducible submission, record the versions that work in your environment.

```text
tensorflow
scikit-learn
pandas
numpy
matplotlib
seaborn
jupyter
```

## Re-running the Notebook

For a clean execution:

1. Open the notebook in Jupyter or VS Code.
2. Restart the kernel to clear old variables.
3. Run all cells from top to bottom.
4. Confirm the dataset path before the loading cell.
5. Check that training and test shapes match the notebook's output.
6. Confirm that the model results table is populated.
7. Confirm that the expected PNG files have been written to `plot/`.
8. Save the notebook after the outputs have finished rendering.
9. Export an HTML copy if the practical submission requires one.

To export with `nbconvert`:

```bash
jupyter nbconvert --to html "IMDB_Sentiment_Analysis.ipynb"
```

If the command is unavailable, install `nbconvert` into the same Python environment as Jupyter.

## Observations

### 1. A baseline is useful

The ANN establishes a reference score before recurrent layers are introduced. It also demonstrates that a strong baseline is valuable when judging whether a more complex model is worth its added cost.

### 2. Sequential structure matters

Recurrent layers update a hidden state across the tokens in a review. That structure is a natural fit for experiments involving word order and contextual dependencies.

### 3. SimpleRNN has limitations on this task

The recorded SimpleRNN model has lower accuracy and recall than the gated alternatives. Its results make the gradient-flow experiments relevant to the overall project.

### 4. Gated architectures improve the recorded results

Both LSTM and GRU outperform SimpleRNN in the recorded metrics. The gates provide additional control over the information carried through the sequence.

### 5. Extra layers cost time

The recorded Stacked LSTM run takes considerably longer per epoch than the other models shown. Additional layers should be justified by measurable benefits, not by complexity alone.

### 6. Metrics answer different questions

Accuracy alone can hide a model's weakness on a particular class. Precision, recall, F1-score, ROC-AUC, confusion matrices, and error examples help build a more complete picture.

### 7. Custom sentences are a useful sanity check

A handful of hand-written examples can expose behaviour that aggregate metrics do not reveal. However, these examples are not a substitute for a separate, representative evaluation dataset.

## Known Limitations

- The model uses a fixed vocabulary and fixed sequence length.
- Words outside the fitted vocabulary are represented using the out-of-vocabulary token.
- Sarcasm, double negation, and mixed sentiment can be difficult to classify.
- A single positive-class threshold may not suit every application.
- Results depend on the data split, initialization, training settings, package versions, and hardware.
- The recorded custom examples are limited and should not be interpreted as a comprehensive language evaluation.
- The project does not use a pretrained transformer or contextual language embedding.
- Training-time comparisons are hardware-dependent and should not be treated as universal benchmarks.
- A model trained on movie reviews may not generalize to product reviews, social-media posts, or other domains without further evaluation.

## Potential Next Steps

- Test additional vocabulary sizes and embedding dimensions.
- Evaluate dropout and recurrent dropout.
- Tune classification thresholds against a stated business objective.
- Inspect calibration of predicted probabilities.
- Expand evaluation for sarcasm, negation, and mixed sentiment.
- Compare against a TF-IDF plus linear-classifier baseline.
- Save the trained model and tokenizer together for consistent inference.
- Build a small interface for entering a review and displaying the prediction.
- Test on a separate dataset from a different source to assess domain shift.
- Record package versions and random seeds for more repeatable experiments.

## Wrap-up

This practical covers the full path from raw review data to a tested sentiment classifier. It includes cleaning, exploratory review-length analysis, target encoding, leakage-aware tokenization, padding, a baseline ANN, recurrent-model training, gradient experiments, and final evaluation.

In the recorded experiment, GRU reaches **89.73% test accuracy**, **89.78% F1-score**, and **0.9595 ROC-AUC**. LSTM records the highest recall at **90.86%**, so the best choice still depends on which errors matter most for the intended use.

The key lesson is that model selection should consider predictive quality, error patterns, parameter count, training cost, and behaviour on different inputs—not just a single score.

---

<div align="center">

### Project notes

**Tools:** Python · TensorFlow/Keras · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter

**Before publishing:** verify all figures, notebook outputs, metric values, and dataset paths against the final version of your own project.

</div>
