# 6. Concatenation Logic in Late Fusion

## 6.1 Concept
In the **Group-wise Late Fusion** architecture, the "Concatenation" step is the critical bridge that merges the processed information from heterogeneous data sources into a single, unified representation.

Unlike "Early Fusion" (which concatenates raw features), our approach first extracts high-level abstractions (embeddings) from each feature group independently using dedicated Encoders.

## 6.2 Mathematical Formulation

Let $E_i$ be the output embedding vector from the encoder for feature group $i$, where $i \in \{\text{Temporal, Weather, Road, Spatial}\}$.

Each encoder is designed to output a fixed-size embedding of dimension $d=64$:
$$ E_{\text{temporal}}, E_{\text{weather}}, E_{\text{road}}, E_{\text{spatial}} \in \mathbb{R}^{64} $$

The **Concatenation Operation** ($\oplus$) joins these vectors along the feature dimension:

$$ Z_{\text{fusion}} = E_{\text{temporal}} \oplus E_{\text{weather}} \oplus E_{\text{road}} \oplus E_{\text{spatial}} $$

## 6.3 Dimensionality
The resulting fused vector $Z_{\text{fusion}}$ has a dimensionality equal to the sum of the individual embedding dimensions:

$$ \text{dim}(Z_{\text{fusion}}) = 64 + 64 + 64 + 64 = \mathbf{256} $$

## 6.4 Role in Architecture
This 256-dimensional vector $Z_{\text{fusion}}$ serves as the input to the final **Fusion MLP**. It contains a balanced representation of the accident context, ensuring that no single feature group (e.g., Weather with 130 raw features) dominates the representation simply due to having more raw inputs than others (e.g., Spatial with only 3).
