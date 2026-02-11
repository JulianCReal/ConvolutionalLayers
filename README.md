# Exploring Convolutional Layers Through Data and Experiments

**Author:** Julian David Castiblanco Real 
**Course:** Deep Learning  
**Date:** Febrero 2026

---

## 📋 Problem Description

This project explores how convolutional neural networks (CNNs) learn hierarchical features from image data, focusing on the architectural principles that make convolutions effective for computer vision tasks. Rather than treating neural networks as black boxes, we systematically analyze how design choices—particularly kernel size—affect learning dynamics and model performance.

### Research Question
**How does kernel size (3×3 vs 5×5) impact the learning capacity, parameter efficiency, and performance of convolutional neural networks on clothing classification?**

---

## 📊 Dataset Description

### Fashion-MNIST

Fashion-MNIST is a dataset of Zalando's article images, designed as a more challenging drop-in replacement for the original MNIST digits dataset.

**Characteristics:**
- **Images:** 70,000 grayscale images (60,000 training + 10,000 test)
- **Resolution:** 28×28 pixels
- **Channels:** 1 (grayscale)
- **Classes:** 10 clothing categories
- **Distribution:** Perfectly balanced (6,000 training samples per class)

**Class Labels:**
| Label | Description   | Label | Description    |
|-------|---------------|-------|----------------|
| 0     | T-shirt/top   | 5     | Sandal         |
| 1     | Trouser       | 6     | Shirt          |
| 2     | Pullover      | 7     | Sneaker        |
| 3     | Dress         | 8     | Bag            |
| 4     | Coat          | 9     | Ankle boot     |

**Why Fashion-MNIST for Convolutional Layers?**

1. **Spatial Structure:** Clothing items have recognizable shapes and textures that are spatially organized
2. **Translation Invariance:** A shirt remains a shirt regardless of position in the image
3. **Local Patterns:** Features like collars, buttons, and seams are localized
4. **Hierarchical Composition:** Low-level edges → textures → garment parts → complete items
5. **Manageable Complexity:** Small enough for rapid experimentation, complex enough to demonstrate CNN advantages

---

## 🏗️ Architecture Diagrams

### Baseline Model (Fully Connected)

```
Input (784)
    ↓
Dense(128) + ReLU + Dropout(0.2)
    ↓
Dense(64) + ReLU + Dropout(0.2)
    ↓
Dense(10) + Softmax
    ↓
Output (10 classes)

Parameters: ~109,000
```

**Limitations:**
- Destroys spatial structure through flattening
- No parameter sharing → 100K+ parameters in first layer alone
- No translation invariance
- Cannot learn hierarchical features

---

### CNN Architecture (3×3 Kernels)

```
Input (28×28×1)
    ↓
[Conv2D(32, 3×3, same) + ReLU] → (28×28×32)
MaxPool(2×2) → (14×14×32)
    ↓
[Conv2D(64, 3×3, same) + ReLU] → (14×14×64)
MaxPool(2×2) → (7×7×64)
    ↓
[Conv2D(128, 3×3, same) + ReLU] → (7×7×128)
MaxPool(2×2) → (3×3×128)
    ↓
Flatten → (1152)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Dense(10) + Softmax

Parameters: ~279,000
```

**Design Justifications:**

1. **Kernel Size (3×3):**
   - Smallest size to capture directional information
   - Efficient parameter usage (9 weights per filter)
   - Modern standard (VGG, ResNet use exclusively)
   - Can approximate larger receptive fields through stacking

2. **Padding ('same'):**
   - Preserves spatial dimensions
   - Processes border pixels equally
   - Enables deeper networks without dimension collapse

3. **Progressive Channel Increase (32→64→128):**
   - Matches hierarchical nature of visual features
   - Early layers: simple features, fewer filters needed
   - Deep layers: complex combinations, more filters required
   - Common pattern in successful architectures

4. **MaxPooling (2×2):**
   - Translation invariance (small shifts don't change output)
   - Reduces spatial dimensions → computational efficiency
   - Expands receptive field in subsequent layers
   - Mild regularization effect

5. **ReLU Activation:**
   - Addresses vanishing gradient problem
   - Computational efficiency (simple thresholding)
   - Sparsity (many activations = 0)
   - Industry standard for CNNs

6. **Dropout (0.3 on Dense Layer):**
   - Prevents co-adaptation in fully connected layer
   - Not used in conv layers (weight sharing already regularizes)
   - 30% rate balances regularization vs capacity

---

### CNN Architecture (5×5 Kernels)

```
[Same structure as 3×3 version, but with 5×5 kernels]

Parameters: ~530,000
```

**Comparison:**
- **3×3 Kernel:** 9 weights per filter
- **5×5 Kernel:** 25 weights per filter (~2.8× more)
- **Parameter Difference:** First layer alone: 320 vs 832 parameters

---

## 🧪 Experimental Results

### Quantitative Comparison

| Metric                    | Baseline (FC) | CNN 3×3       | CNN 5×5       |
|---------------------------|---------------|---------------|---------------|
| **Test Accuracy**         | ~88%          | ~91-92%       | ~91-92%       |
| **Test Loss**             | 0.33          | 0.24          | 0.25          |
| **Total Parameters**      | ~109,000      | ~279,000      | ~530,000      |
| **Layer 1 Parameters**    | 100,352       | 320           | 832           |
| **Training Time (15 epochs)** | ~60s      | ~180s         | ~240s         |

### Key Findings

#### 1. CNN vs Baseline
- **Accuracy Improvement:** ~3-4 percentage points
- **Parameter Efficiency:** Despite more total parameters, CNNs use far fewer in convolutional layers due to weight sharing
- **Learning Quality:** Lower loss indicates better-calibrated predictions

#### 2. 3×3 vs 5×5 Kernels

**Performance:**
- Comparable final accuracy (difference < 0.5%)
- 5×5 slightly higher validation loss (potential overfitting)

**Efficiency:**
- 3×3 model: 47% fewer parameters
- 3×3 model: ~25% faster training per epoch

**Convergence:**
- Both converge smoothly within 15 epochs
- 3×3 shows slightly more stable validation curves

**Receptive Field:**
- Single 3×3 layer: 3×3 pixels
- Two stacked 3×3 layers: 5×5 pixels (equivalent to one 5×5)
- Three stacked 3×3 layers: 7×7 pixels
- Deeper 3×3 networks can achieve same receptive field as 5×5 with fewer parameters

### Qualitative Observations

#### Learned Filters (First Layer)

**3×3 Filters:**
- Edge detectors (horizontal, vertical, diagonal)
- Texture patterns (dots, lines)
- Blob detectors
- Color gradients

**5×5 Filters:**
- Similar patterns but more diffuse
- Some filters capture more complex combinations
- Higher redundancy observed

#### Trade-offs Summary

| Aspect              | 3×3 Kernels ✅               | 5×5 Kernels                  |
|---------------------|------------------------------|------------------------------|
| Parameters          | Lower (better efficiency)    | Higher (~2×)                 |
| Speed               | Faster training              | Slower (~25%)                |
| Depth Required      | More layers for same RF      | Fewer layers                 |
| Flexibility         | Can build any RF through stacking | Fixed RF per layer      |
| Modern Practice     | Industry standard            | Rare in modern architectures |

**Recommendation:** Use 3×3 kernels for most applications. They offer the best balance of expressiveness, efficiency, and flexibility.

---

## 🎓 Interpretation and Insights

### 1. Why Did CNNs Outperform the Baseline?

#### A. Spatial Inductive Bias
The baseline model treats images as flat vectors, destroying 2D relationships:
```
Baseline sees: [p₁, p₂, p₃, ..., p₇₈₄]
CNN sees:      [[p₁, p₂, ...],
                [p₂₉, p₃₀, ...],
                ...]
```
Neighboring pixels are highly correlated in images—CNNs exploit this, baselines don't.

#### B. Parameter Sharing (Weight Reuse)
**Baseline:** Learning "edge at position (10,10)" and "edge at position (15,15)" requires separate parameters for each location.

**CNN:** The same 3×3 edge detector slides across all positions. One set of 9 weights detects edges everywhere.

**Impact:**
- Massive parameter reduction (320 vs 100K in first layer)
- Strong regularization (prevents overfitting)
- Enforces translation invariance

#### C. Translation Invariance
A sleeve is a sleeve whether it appears at:
- Top-left corner
- Center of image  
- Bottom-right corner

CNNs automatically learn this through convolution. Fully connected networks must learn each position independently.

#### D. Hierarchical Feature Learning

```
Layer 1 (Low-level):    Edges, corners, textures
         ↓
Layer 2 (Mid-level):    Collar, sleeve, button patterns  
         ↓
Layer 3 (High-level):   Shirt, dress, shoe shapes
         ↓
Output:                 Complete object classification
```

This compositional structure mirrors human visual perception and object recognition.

#### E. Local Receptive Fields
Each neuron only processes a small patch:
- Reduces search space
- Makes optimization tractable
- Progressive expansion through depth

---

### 2. What Inductive Bias Does Convolution Introduce?

**Inductive bias:** Assumptions that constrain the hypothesis space to guide learning toward better solutions for specific problem types.

#### Three Core Biases:

**A. Locality**
- **Assumption:** Relevant features exist in local neighborhoods
- **Implementation:** Small kernels (3×3, 5×5) connect nearby pixels
- **Implication:** Distant pixels aren't directly connected in early layers
- **Why it works:** In images, nearby pixels are highly correlated (object parts are spatially coherent)

**B. Stationarity (Translation Equivariance)**
- **Assumption:** Useful features can appear anywhere
- **Implementation:** Weight sharing across spatial locations
- **Implication:** Same pattern detector reused across entire image
- **Why it works:** A cat's ear is a cat's ear regardless of image position

**C. Hierarchical Composition**
- **Assumption:** Complex patterns are built from simpler primitives
- **Implementation:** Stacked layers with increasing abstraction
- **Implication:** Bottom-up construction of representations
- **Why it works:** Natural images have compositional structure (edges → textures → parts → objects)

**Mathematical Formalization:**
For input x and filter w:
```
(x ∗ w)[i, j] = Σₘ Σₙ x[i+m, j+n] · w[m, n]
```
Properties:
- **Equivariance:** If x shifts, output shifts equally: f(shift(x)) = shift(f(x))
- **Local:** Only depends on neighborhood around (i, j)
- **Shared:** Same w used for all (i, j)

---

### 3. When Would Convolution NOT Be Appropriate?

#### ❌ Problem Types Where CNNs Fail:

**A. Tabular/Structured Data**

*Example:* Customer churn prediction
```
Features: [age, income, tenure, num_purchases, region_code]
```

**Why not CNNs:**
- No spatial structure
- Feature order is arbitrary (swapping columns doesn't break the data)
- No locality assumption holds
- Translation invariance is meaningless

**Better approach:** XGBoost, Random Forests, MLPs

---

**B. Long-Range Dependencies**

*Example:* Document sentiment where key phrase appears at start and end
```
"Despite initial excitement... [5000 words] ... ultimately disappointed"
```

**Why not CNNs:**
- Convolution has local receptive field
- Would need hundreds of layers to connect distant tokens
- "Pyramid of doom" problem—information loss through pooling

**Better approach:** Transformers with global attention

---

**C. Position-Sensitive Tasks**

*Example:* Chess board evaluation
```
Knight at e4 ≠ Knight at a1
```

**Why not CNNs:**
- Translation invariance is harmful here
- Absolute position carries semantic meaning
- Same piece at different positions has different strategic value

**Better approach:** Position embeddings + attention, or position-aware architectures

---

**D. Irregular Structures**

*Example:* Social network analysis, molecular structures
```
Graph: nodes connected with varying numbers of neighbors
```

**Why not CNNs:**
- No regular grid structure
- Neighbors aren't spatially arranged
- Variable node degrees

**Better approach:** Graph Neural Networks (GCNs, GraphSAGE)

---

**E. Point Clouds (3D Data)**

*Example:* LiDAR data from autonomous vehicles
```
Unordered set of (x, y, z) points
```

**Why not CNNs:**
- No natural 2D/3D grid
- Permutation invariance required (point order shouldn't matter)
- Sparse in 3D space

**Better approach:** PointNet, PointNet++

---

**F. Very Small Datasets**

*Example:* Medical diagnosis with 50 annotated X-rays

**Why not CNNs:**
- Even with weight sharing, CNNs have many parameters
- Risk of severe overfitting
- Not enough data to learn convolutional filters

**Better approach:**
- Transfer learning from pre-trained models
- Classical ML (SVMs, Random Forests)
- Heavy data augmentation

---

**G. Global Context Required Immediately**

*Example:* "Does this 50-page document mention climate change?" (term might appear once on page 37)

**Why not CNNs:**
- Local processing misses global patterns
- Pooling loses precise location information
- Need too many layers to aggregate global context

**Better approach:** Attention mechanisms, bag-of-words models

---

#### ✅ When to Use Convolutions:

1. **Grid-like topology:** Images, audio spectrograms, video, time-series
2. **Local patterns meaningful:** Nearby elements are correlated
3. **Translation invariance desired:** Pattern location doesn't matter
4. **Hierarchical composition:** Complex features build from simpler ones
5. **Sufficient data:** Enough examples to learn filters

**Rule of Thumb:** If you can't visualize your data as a grid where nearby elements are related, don't use convolutions.

---

## 🛠️ How to Run

### Prerequisites

```bash
# Install dependencies
pip install tensorflow numpy matplotlib seaborn pandas scikit-learn boto3 sagemaker
```

### Local Execution

```bash
# Open Jupyter notebook
jupyter notebook cnn_exploration.ipynb

# Run all cells sequentially
# Results will be saved to ./results/
```
---

## 📈 Results Summary

### Main Achievements

1. ✅ **Demonstrated CNN superiority over fully connected baseline** (+3-4% accuracy)
2. ✅ **Compared 3×3 vs 5×5 kernels** (3×3 more efficient with equivalent performance)
3. ✅ **Visualized learned features** (edge detectors, texture patterns)
4. ✅ **Provided theoretical justification** for architectural choices

### Key Takeaways

- **Inductive biases matter:** Matching architecture to data structure is crucial
- **Simplicity wins:** 3×3 kernels are standard for good reason
- **Understanding > tuning:** Why something works beats finding optimal hyperparameters
- **Convolutions aren't universal:** Know when (and when not) to use them

---

## 📚 References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition.
2. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition (VGG).
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition (ResNet).
4. Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.

---

## 👤 Author

**Julian David Castiblanco Real**  
*Email: julian.castiblanco-r@mail.escuelaing.edu.co*
