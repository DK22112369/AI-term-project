# Figure 4: Model Architecture Diagram (Mermaid.js)

You can copy the code below and paste it into [Mermaid Live Editor](https://mermaid.live/) to generate the image, or use it directly in tools that support Mermaid.

```mermaid
graph TD
    subgraph Inputs
        I1[Temporal Features<br/>(14 dims)]
        I2[Weather Features<br/>(130 dims)]
        I3[Road Features<br/>(12 dims)]
        I4[Spatial Features<br/>(3 dims)]
    end

    subgraph Encoders [Group-wise Encoders]
        E1[MLP Encoder<br/>(64 units)]
        E2[MLP Encoder<br/>(64 units)]
        E3[MLP Encoder<br/>(64 units)]
        E4[MLP Encoder<br/>(64 units)]
    end

    subgraph Fusion [Late Fusion Stage]
        C[Concatenation<br/>(256 dims)]
        F1[Fusion Layer 1<br/>(128 units)]
        F2[Fusion Layer 2<br/>(64 units)]
        Out[Output Layer<br/>(4 Classes)]
    end

    %% Connections
    I1 --> E1
    I2 --> E2
    I3 --> E3
    I4 --> E4

    E1 --> C
    E2 --> C
    E3 --> C
    E4 --> C

    C --> F1
    F1 --> F2
    F2 --> Out

    %% Styling
    style Inputs fill:#f9f,stroke:#333,stroke-width:2px
    style Encoders fill:#bbf,stroke:#333,stroke-width:2px
    style Fusion fill:#bfb,stroke:#333,stroke-width:2px
    style Out fill:#f66,stroke:#333,stroke-width:4px,color:white
```
