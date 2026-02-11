# 📜 LSTM Text Generator (Shakespeare)

A character-level LSTM-based text generation model built using
TensorFlow/Keras and trained on the Complete Works of William
Shakespeare.

This project follows a clearly structured 16-step deep learning
pipeline, from data collection to model saving.

------------------------------------------------------------------------

# 🔄 Project Workflow (Step-by-Step)

## STEP 1: Importing Libraries

-   TensorFlow / Keras
-   NumPy
-   Scikit-learn
-   Requests
-   Pickle
-   Regex utilities

------------------------------------------------------------------------

## STEP 2: Setting Random Seeds

-   `np.random.seed(42)`
-   `tf.random.set_seed(42)` Ensures reproducible training results.

------------------------------------------------------------------------

## STEP 3: Downloading Shakespeare Dataset

-   Downloads dataset from Project Gutenberg\
-   Includes fallback sample text if download fails

Source: https://www.gutenberg.org/files/100/100-0.txt

------------------------------------------------------------------------

## STEP 4: Preprocessing Text Data

Text cleaning pipeline: 1. Convert to lowercase\
2. Remove special characters\
3. Normalize whitespace

Dataset trimmed to 200,000 characters for efficient training.

------------------------------------------------------------------------

## STEP 5: Creating Character Vocabulary

-   Extract unique characters\
-   Create:
    -   `char_to_idx`
    -   `idx_to_char`
-   Determine vocabulary size

------------------------------------------------------------------------

## STEP 6: Creating Training Sequences

-   Sequence Length = 40
-   Step Size = 3
-   Creates input-output character pairs

------------------------------------------------------------------------

## STEP 7: Encoding Sequences to Numbers

-   Convert characters to integer indices\
-   One-hot encode target outputs\
-   Prepare `X` and `y` arrays

------------------------------------------------------------------------

## STEP 8: Splitting Data (80% Train, 20% Validation)

-   Uses `train_test_split`
-   Ensures reproducible split

------------------------------------------------------------------------

## STEP 9: Building LSTM Model Architecture

Model Architecture:

1.  Embedding Layer (128 dimensions)
2.  LSTM (256 units, return_sequences=True)
3.  LSTM (256 units)
4.  Dropout Layers
5.  Dense (128, ReLU)
6.  Output Dense (Softmax)

------------------------------------------------------------------------

## STEP 10: Compiling the Model

-   Optimizer: Adam (learning_rate=0.001)
-   Loss: Categorical Crossentropy
-   Metric: Accuracy

------------------------------------------------------------------------

## STEP 11: Configuring Training Callbacks

-   EarlyStopping (patience=5)
-   ModelCheckpoint (saves best model)

------------------------------------------------------------------------

## STEP 12: Training the Model

-   Batch Size: 256
-   Epochs: 100 (with early stopping)
-   Tracks training time and performance

------------------------------------------------------------------------

## STEP 13: Evaluating Model Performance

-   Training Accuracy
-   Validation Accuracy
-   Performance requirement checks:
    -   ≥ 90% accuracy
    -   Training time \< 60 minutes

------------------------------------------------------------------------

## STEP 14: Creating Text Generation Function

Implements temperature-based sampling.

Function:

``` python
generate_text(model, seed_text, length=200, temperature=0.7)
```

Temperature Controls Creativity: - 0.5 → Conservative - 0.7 → Balanced -
1.0 → Creative

------------------------------------------------------------------------

## STEP 15: Generating Sample Texts

Generates text using sample seeds: - "to be or not to be" - "shall i
compare thee to" - "what light through yonder"

Tested with multiple temperature values.

------------------------------------------------------------------------

## STEP 16: Saving Model and Metadata

Saves: - `lstm_text_generator.keras` - `model_metadata.pkl` -
`training_history.pkl`

Metadata includes vocabulary mappings and configuration.

------------------------------------------------------------------------

# 📂 Project Structure

    lstm-text-generator/
    │
    ├── data/
    ├── models/
    ├── outputs/
    ├── src/
    ├── untitled0.ipynb
    ├── lstm_text_generator.keras
    ├── model_metadata.pkl
    ├── training_history.pkl
    └── README.md

------------------------------------------------------------------------

# 🛠 Technologies Used

-   Python
-   TensorFlow / Keras
-   NumPy
-   Scikit-learn
-   Requests

------------------------------------------------------------------------

# 🎯 Project Objective

To build a high-accuracy LSTM-based character-level text
generator capable of producing Shakespeare-style text using deep
learning techniques.

------------------------------------------------------------------------

# 👨‍💻 Author

Deep Learning LSTM Text Generation Project
