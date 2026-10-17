# Autoregressive Text Generation Using Recurrent Neural Networks

**Course:** Foundations of Generative AI (ICT4403)  
**Assignment:** Assignment 2

## 1. Problem Statement

This project develops an autoregressive text-generation system using four recurrent neural network architectures: Simple RNN, Bidirectional RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU). The models are trained to predict the next word from a sequence of preceding words and generate text from seed sequences.

## 2. Objectives

- Prepare a textual dataset for next-word prediction.
- Implement and train four recurrent neural network models.
- Compare models using test loss, perplexity, prediction accuracy, and training time.
- Generate text using different seed sequences and sampling temperatures.
- Analyse generated text for coherence, diversity, contextual consistency, and repetition.

## 3. Dataset

**Dataset:** Tiny Shakespeare

The dataset contains Shakespearean text, including dialogue, character names, and dramatic language.

**Dataset source:** [Tiny Shakespeare in Karpathy's char-rnn repository](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). The script downloads it when run.

The text is preprocessed and converted into input-target sequences for next-word prediction. The model uses an input sequence to predict the corresponding next word.

## 4. Models Implemented

- **Simple RNN:** A recurrent neural network used as the baseline model.
- **Bidirectional RNN:** Processes sequences in forward and backward directions during training.
- **LSTM:** Uses memory cells and gates to learn sequential dependencies.
- **GRU:** Uses update and reset gates to learn sequential patterns with a simpler gated architecture than LSTM.

All four models are evaluated under the experimental conditions used by the implementation.

## 5. Evaluation Metrics

- **Test loss:** Measures the model's prediction error.
- **Perplexity:** Measures uncertainty in next-word prediction; lower values are better.
- **Prediction accuracy:** Measures the proportion of correct next-word predictions.
- **Training time:** Measures the time required to train each model.
- **Qualitative evaluation:** Examines coherence, diversity, contextual consistency, and repetition in generated text.

## 6. Experimental Results

| Model | Test Loss | Test Perplexity | Test Accuracy | Training Time (s) |
|---|---:|---:|---:|---:|
| Simple RNN | 5.1404 | 170.78 | 16.27% | 102.13 |
| Bidirectional RNN | 5.2347 | 187.68 | 15.51% | 187.42 |
| LSTM | 5.1001 | 164.04 | 14.39% | 111.99 |
| GRU | 5.0906 | 162.49 | 13.85% | 138.11 |

The GRU achieved the lowest test loss and perplexity. The Simple RNN achieved the highest next-word prediction accuracy and the shortest training time. The Bidirectional RNN had the highest test perplexity and the longest training time.

## 7. Sampling Temperature Analysis

Text was generated from the same seed sequences at temperatures 0.5, 1.0, and 1.5.

- **Temperature 0.5:** Produces more conservative predictions. Repetition was observed in some generated samples.
- **Temperature 1.0:** Produces more varied word choices, although coherence remains limited in some samples.
- **Temperature 1.5:** Produces more unpredictable text, but increased variety does not guarantee meaningful or contextually consistent output.

Some generated samples contain unknown-token markers such as `<unk>`. This is a limitation of the current vocabulary and preprocessing setup.

The measured bigram repetition ratios at temperature 0.5 were 0.071 for Simple RNN, 0.180 for Bidirectional RNN, 0.142 for LSTM, and 0.407 for GRU. This indicates greater adjacent-bigram repetition for GRU at this temperature in the current experiment.

## 8. Conclusion

The GRU was the best-performing model in terms of test loss and perplexity. The Simple RNN achieved the highest next-word prediction accuracy and was the fastest model to train. The LSTM provided competitive predictive performance, while the Bidirectional RNN required the longest training time and had the highest perplexity.

Therefore, the GRU is selected as the best model according to loss and perplexity. However, the generated samples indicate that predictive performance alone does not establish superior text coherence. Both quantitative metrics and qualitative analysis are important when evaluating autoregressive text generation.

## 9. Project Structure

The repository contains the Python implementation, supporting configuration files, experimental results, generated text samples, and plots.

## 10. Installation and Execution

1. Install Python and the dependencies listed in `requirements.txt`.
2. Open a terminal in the project directory.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Python source file:

   ```bash
   python autoregressive_text_generation.py
   ```

   Replace the filename above if your implementation uses a different filename.

5. Inspect the generated CSV files, summary JSON, plots, and text samples in the results directory.
