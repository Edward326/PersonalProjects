# AI / Deep Learning Experiments

This part of the `AI` folder collects five standalone PyTorch scripts, each implementing a
different model architecture end-to-end: data loading, network definition, training loop and
evaluation/inference. They were built as individual learning exercises covering classic CNNs,
attention mechanisms, sequence-to-sequence transformers and multi-modal encoder-decoder models.
Each script is self-contained and can be run directly with `python <script>.py`.

| Project | Task | Core architecture | Dataset |
|---|---|---|---|
| [ICaptModel_NeuralICaptModel_FLICKRDATASET](#icaptmodel_neuralicaptmodel_flickrdataset) | Image captioning | CNN encoder + Multi-Head Attention + Bahdanau Attention decoder | Flickr8k |
| [LeNet](#lenet) | Digit/image classification | LeNet-5 (classic CNN) | SVHN |
| [ResNet18](#resnet18) | Image classification | Custom ResNet-18 (residual blocks) | SVHN |
| [SemanticSegm_ODetection](#semanticsegm_odetection) | Semantic segmentation | Fully Convolutional Network on top of ResNet-18 | Pascal VOC 2012 |
| [TransformerBased_Translator](#transformerbased_translator) | Machine translation | Transformer (encoder-decoder, multi-head attention) | English-French (fra-eng) |

All scripts share the same general pattern:
- Auto-download and cache the required dataset on first run.
- Build a `DataLoader`-based training/validation/test pipeline.
- Define the model as one or more `nn.Module` classes.
- Run a training loop that reports loss/accuracy per epoch, times the run, and plots the results
  with `matplotlib`.
- Save the learned weights to a `.pth` file, and support a "load pretrained weights and only
  evaluate" mode via an `alreadyParameters` / mode flag, so a script can be re-run for inference
  without retraining.

---

## ICaptModel_NeuralICaptModel_FLICKRDATASET

**File:** `main_2att_layer.py`

### Purpose
Implements an image captioning system in the spirit of the "Show and Tell" / Neural Image
Caption (NIC) family of models: given a photo, the model generates a natural-language sentence
describing it. It is trained and evaluated on the Flickr8k dataset, which pairs ~8,000 images
with five human-written captions each.

### Implementation
The model follows an encoder-decoder design with two stacked attention mechanisms rather than a
single one:

- **Encoder (`EncoderCNN`)** — an Inception-v3 CNN, pretrained on ImageNet, with its final
  classification layer replaced by a linear projection into the embedding space. Only this
  projection layer is left trainable; the rest of the backbone stays frozen so training only
  updates a small number of parameters.
- **Decoder (`DecoderWithAttention`)** — combines two attention layers in sequence:
  1. A **Multi-Head Attention** layer (`MultiHeadAttention`, implemented from scratch with
     scaled dot-product attention) that attends over the single image embedding coming from the
     encoder and produces a context vector.
  2. That context is used to seed a **Bahdanau-style additive attention** sequence-to-sequence
     decoder (`Seq2SeqEncoder` + `Seq2SeqAttentionDecoder`, GRU-based) that consumes the caption
     token embeddings and outputs a probability distribution over the vocabulary at every step.
- **Vocabulary** — built from scratch (`Vocabulary` class) from the training captions using
  spaCy tokenization, with a configurable minimum-frequency cutoff and `<PAD>`/`<BOS>`/`<EOS>`/
  `<UNK>` special tokens.
- **Data pipeline** — `FlickrDataset` (a `torch.utils.data.Dataset`) plus a custom
  `CollateDataset` collate function that pads variable-length captions per batch.
- **Inference (`caption_image`)** — greedy autoregressive decoding: the model starts from
  `<BOS>` and repeatedly feeds its own previous prediction back in until it emits `<EOS>` or
  hits a maximum caption length.
- **Training loop** — supports both a "train from scratch" mode and an `already_learned` mode
  that reloads saved weights and runs a single evaluation pass on the test set, reporting
  average loss, epoch timings and an ETA while training.

Datasets and vocabulary text files (`train.txt`, `validation.txt`, `test.txt`) are generated
automatically from the raw Flickr8k annotation files on first run.

---

## LeNet

**File:** `ex2.py`

### Purpose
A from-scratch training pipeline for the classic **LeNet-5** convolutional network, used here as
a baseline image classifier on the Street View House Numbers (SVHN) dataset (10-digit
classification).

### Implementation
- **Network** — a plain `nn.Sequential` stack mirroring the original LeNet-5 architecture:
  two convolution + sigmoid + average-pooling blocks, followed by three fully connected layers
  ending in a 10-way output (`Conv2d(3,6) → Conv2d(6,16) → FC(120) → FC(84) → FC(10)`).
- **Weight initialization** — Xavier/Glorot uniform initialization applied to all linear and
  convolutional layers.
- **Data** — `torchvision.datasets.SVHN`, downloaded automatically and split into
  train/validation/test `DataLoader`s.
- **Training loop** — plain SGD optimization with cross-entropy loss; each epoch reports
  training and validation loss/accuracy, and the run is timed end-to-end.
- **Persistence** — model weights are saved to `ex2__LeNet_parameters.pth` after training, and
  can be reloaded directly for test-only evaluation via the `already_learned` flag, skipping
  training entirely.
- **Reporting** — loss and accuracy curves (train vs. validation) are plotted with `matplotlib`
  at the end of a training run.

---

## ResNet18

**Files:** `resnet.py`, `test.py`

### Purpose
Trains a **ResNet-18-style** convolutional network from scratch (not the pretrained
`torchvision` version) on the SVHN dataset, to compare residual-connection based classification
against the plain LeNet baseline above.

### Implementation
- **Residual block (`ResBlock`)** — implemented manually: two stacked `Conv2d → BatchNorm →
  ReLU` layers, with a shortcut identity connection. When the number of channels changes between
  stages, a `1x1` convolution projects the input so it can still be added to the block's output
  (the standard ResNet "projection shortcut").
- **Network** — built as an `nn.Sequential` stem (`Conv2d(3,64,k=7,s=2) → BatchNorm → MaxPool`)
  followed by four stages of residual blocks with increasing channel width
  (64 → 128 → 256 → 512), each stage's channel expansion handled by a `1x1` convolution between
  blocks, ending in global average pooling and a linear classifier head.
- **Training/evaluation** — structurally identical to the LeNet script (same
  `try_gpu`/`train_epoch`/`evaluate_accuracy`/`train` helpers), so both models can be compared
  under the same experimental conditions. Weights are checkpointed to
  `ResNet_<epochs>parameters.pth`, and an `already_learned` mode allows reloading a trained model
  to run test-set evaluation only.
- **Reporting** — same loss/accuracy plotting behavior as the LeNet script.

---

## SemanticSegm_ODetection

**File:** `Model/main.py`

### Purpose
A **Fully Convolutional Network (FCN)** for pixel-wise semantic segmentation, trained on the
Pascal VOC 2012 segmentation dataset (21 classes, including background). Given an image, the
model predicts a class label for every pixel rather than a single label for the whole image.

### Implementation
- **Backbone** — a pretrained `torchvision` ResNet-18 with its final average-pooling and fully
  connected layers stripped off, used purely as a feature extractor.
- **Segmentation head** — a `1x1` convolution maps the backbone's feature channels down to the
  21 VOC classes, followed by a transposed convolution (`ConvTranspose2d`) that upsamples the
  coarse feature map back to (approximately) the original image resolution. The transposed
  convolution is explicitly initialized with a **bilinear interpolation kernel**
  (`bilinear_kernel`) rather than random weights, which is a known technique to stabilize FCN
  training from the start.
- **Label handling** — VOC segmentation masks are stored as RGB images where each color encodes
  a class; `voc_colormap2label` / `voc_label_indices` convert that colormap encoding into
  per-pixel integer class indices usable by a cross-entropy loss. `VOC_COLORMAP` and
  `VOC_CLASSES` enumerate the 21 supported classes (background, person, car, cat, etc.).
- **Data pipeline (`VOCSegDataset`)** — loads and normalizes images with ImageNet statistics,
  filters out images smaller than the training crop size, and applies synchronized random
  cropping to image/label pairs (`voc_rand_crop`) for data augmentation.
- **Loss/metric** — per-pixel cross-entropy loss (averaged spatially), and pixel accuracy
  computed after bilinearly resizing predictions back to the label resolution.
- **Inference helpers** — `predict` runs the trained network on a single image, and
  `label2image` converts the predicted class-index map back into an RGB visualization using the
  VOC colormap, for qualitative inspection of segmentation results.
- **Training loop** — reports per-epoch training loss/accuracy and test accuracy, and plots all
  three curves together at the end.

---

## TransformerBased_Translator

**File:** `main/main.py`

### Purpose
An English→French neural machine translation system built as a **Transformer** encoder-decoder
from first principles (no `nn.Transformer` — every component is implemented by hand), trained on
the `fra-eng` parallel sentence dataset.

### Implementation
- **Tokenization/vocabulary** — a custom `Vocab` class builds separate source (English) and
  target (French) vocabularies from whitespace-tokenized, lower-cased text, with
  `<pad>`/`<bos>`/`<eos>` reserved tokens and a minimum-frequency cutoff to control vocabulary
  size.
- **Positional encoding** — standard sinusoidal position embeddings (`PositionalEncoding`),
  added to token embeddings before they enter the encoder/decoder stacks.
- **Attention** — a from-scratch scaled dot-product `MultiHeadAttention` module (queries, keys
  and values are split across heads via `transpose_qkv`/`transpose_output`), used both for
  self-attention and cross-attention.
- **Encoder** — a stack of `EncoderBlock`s, each combining multi-head self-attention, a
  position-wise feed-forward network (`PositionWiseFFN`), and residual connections with layer
  normalization (`AddNorm`) after each sub-layer — the standard "Attention Is All You Need"
  layout.
- **Decoder** — a stack of `DecoderBlock`s with masked self-attention over previously generated
  tokens (using a validity-length mask so a token cannot attend to future positions during
  training), followed by encoder-decoder cross-attention and a feed-forward sub-layer, again with
  residual + layer-norm around each stage. During autoregressive inference, each decoder block
  caches its previous key/value states (`state[2][self.i]`) so generation only needs to process
  one new token at a time.
- **Training (`train_seq2seq`, referenced from the shared training utilities)** — teacher forcing
  is used during training (the decoder is fed the ground-truth target sequence shifted by one
  position), with a masked loss that ignores `<pad>` positions.
- **Inference (`predict_seq2seq`)** — greedy decoding: starts from `<bos>` and feeds each
  predicted token back into the decoder until `<eos>` is produced or the step limit is reached.
  Optionally records attention weights for later visualization.
- **Evaluation (`bleu`)** — implements the BLEU score metric from scratch (n-gram precision with
  a brevity penalty) to quantitatively evaluate translation quality.
- **Interactive use** — after training, the script prompts the user on the command line for an
  English sentence and prints the model's French translation.

---

## Notes

- These scripts were written as individual experiments rather than a shared library, so some
  utility functions (`try_gpu`, `download`, `download_extract`, plotting helpers, etc.) are
  duplicated across files rather than factored into a common module.
- Several scripts assume a CUDA-capable GPU is available for reasonable training times, but all
  fall back to CPU automatically via `try_gpu()` if no GPU is found.
- Required third-party packages across the five scripts: `torch`, `torchvision`, `matplotlib`,
  `numpy`, `pandas`, `Pillow`, `requests`, `spacy` (with the `en_core_web_sm` model), and
  `colorama`.
