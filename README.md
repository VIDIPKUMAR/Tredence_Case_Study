# Tredence_Case_Study
# Self-Pruning Neural Network using Learnable Gates

## Overview
This project implements a self-pruning neural network using a custom **PrunableLinear** layer. The model learns both the weights and which connections are important, allowing unnecessary connections to be automatically pruned during training.

Each weight is paired with a learnable gate that decides whether the connection remains active or gets suppressed.

---

## 1. PrunableLinear Layer

Instead of a standard linear layer, this project uses a custom layer where every weight has a corresponding gate value:

\[
output = (W \odot \sigma(G)) \cdot x + b
\]

Where:

- **W** = weight matrix  
- **G** = learnable gate parameters  
- **σ(G)** = sigmoid function (values between 0 and 1)  
- **⊙** = element-wise multiplication  

### Key Idea

- Gate close to **1** → connection stays active  
- Gate close to **0** → connection is effectively removed  

This enables the network to learn which connections are useful.

---

## 2. Sparsity Regularization (L1 Loss)

To encourage pruning, an L1 penalty is applied to the gate values:

\[
Total\ Loss = CrossEntropyLoss + \lambda \sum gates
\]

### Why L1 Regularization?

- Pushes small values toward zero  
- Encourages sparse connections  
- Helps remove unnecessary weights  

Unlike L2 regularization, L1 is more effective for pruning.

---

## 3. Training Setup

- **Dataset:** CIFAR-10  
- **Model:** Fully Connected Neural Network with PrunableLinear layers  
- **Optimizer:** Adam  
- **Loss Function:** CrossEntropy + λ × Sparsity Loss  
- **Epochs:** 10  

---

## 4. Evaluation Metrics

### Test Accuracy
Measures performance on unseen test data.

### Sparsity Level

\[
Sparsity = \frac{Gates < 10^{-2}}{Total\ Gates} \times 100
\]

This shows how many connections were effectively pruned.

---

## 5. Experimental Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|-----------|------------------|-------------|
| 1e-6      | 55.97            | 0.01        |
| 1e-5      | 55.71            | 0.43        |
| 1e-4      | 56.16            | 1.45        |

---

## 6. Analysis of λ Trade-off

The value of **λ** controls the balance between:

- Accuracy  
- Model Compression  

### Observations

#### λ = 1e-6
- Very weak pruning pressure  
- Model behaves like a normal neural network  
- Almost no sparsity  

#### λ = 1e-5
- Slight increase in pruning  
- Accuracy remains stable  

#### λ = 1e-4
- Better sparsity achieved  
- Accuracy still maintained  

### Key Insight

Increasing λ increases pruning strength, but the current values still produce limited sparsity.

---

## 7. Gate Distribution

Histogram analysis of gate values helps understand connection importance:

- Values near **0** → pruned connections  
- Higher values → useful connections  

Most gates remain away from zero, matching the low sparsity results.

---

## 8. Differentiable Pruning Mechanism

The pruning process is fully differentiable.

Gradients flow through:

- Weight matrix **W**  
- Gate parameters **G**

Since sigmoid is smooth, no custom backward pass is needed.

The network learns:

- What to learn (weights)  
- What to prune (gates)

---

## 9. Conclusion

This project demonstrates a simple and effective self-pruning neural network using learnable gates.

### Key Takeaways

- Custom **PrunableLinear** layer enables gating  
- L1 regularization promotes sparsity  
- Model can prune itself during training  
- λ controls the sparsity vs accuracy trade-off  

Although current pruning is limited, the framework is scalable and can be improved further.

---

## 10. Future Improvements

- Use larger λ values for stronger pruning  
- Replace dense layers with convolutional layers  
- Apply structured pruning (neurons / channels)  
- Fine-tune after pruning for better accuracy  

---

## Final Remark

This project provides a strong foundation for efficient deep learning, model compression, and adaptive neural network design.
