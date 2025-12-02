# UrbanMamba Architecture Diagram

Below is a Mermaid diagram representing the dual-branch UrbanMamba segmentation model as implemented in `semanticsegmentation/models/UrbanMamba.py`.

```mermaid
flowchart TD
    %% Styling
    classDef inputStyle fill:#e1f5ff,stroke:#0288d1,stroke-width:3px,color:#000
    classDef spatialStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef waveletStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef fusionStyle fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    classDef decoderStyle fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#000
    classDef outputStyle fill:#ffebee,stroke:#d32f2f,stroke-width:3px,color:#000
    classDef processStyle fill:#e0e0e0,stroke:#424242,stroke-width:2px,color:#000

    %% ============ INPUT ============
    INPUT["🖼️ Input Image<br/>(B, C, H, W)"]:::inputStyle
    
    %% ============ DUAL ENCODER BRANCHES ============
    
    %% Spatial Branch
    subgraph SPATIAL["🔷 SPATIAL ENCODER BRANCH"]
        direction TB
        SE["Backbone_VSSM<br/>(Vision Mamba)"]:::spatialStyle
        S0["Stage 0<br/>C₀ channels"]:::spatialStyle
        S1["Stage 1<br/>C₁ channels"]:::spatialStyle
        S2["Stage 2<br/>C₂ channels"]:::spatialStyle
        S3["Stage 3<br/>C₃ channels"]:::spatialStyle
        
        SE --> S0 --> S1 --> S2 --> S3
    end
    
    %% Wavelet Branch
    subgraph WAVELET["🔶 WAVELET ENCODER BRANCH"]
        direction TB
        HAAR["2D Haar Wavelet<br/>Decomposition"]:::processStyle
        
        subgraph SUBBANDS["Subband Processing"]
            direction LR
            LL["LL<br/>(Low-Low)"]:::waveletStyle
            LH["LH<br/>(Low-High)"]:::waveletStyle
            HL["HL<br/>(High-Low)"]:::waveletStyle
            HH["HH<br/>(High-High)"]:::waveletStyle
        end
        
        ENC["4× Independent<br/>Backbone_VSSM<br/>(deep copies)"]:::waveletStyle
        
        W0["W0: Concat[LL,LH,HL,HH]<br/>4×C₀ channels"]:::waveletStyle
        W1["W1: Concat[LL,LH,HL,HH]<br/>4×C₁ channels"]:::waveletStyle
        W2["W2: Concat[LL,LH,HL,HH]<br/>4×C₂ channels"]:::waveletStyle
        W3["W3: Concat[LL,LH,HL,HH]<br/>4×C₃ channels"]:::waveletStyle
        
        HAAR --> SUBBANDS
        SUBBANDS --> ENC
        ENC --> W0 --> W1 --> W2 --> W3
    end
    
    %% ============ FUSION LAYER ============
    subgraph FUSION["🔗 STAGE-WISE FUSION MODULES"]
        direction TB
        F0["Fusion Module 0<br/>Proj + Concat + Reduce → Cₓ"]:::fusionStyle
        F1["Fusion Module 1<br/>Proj + Concat + Reduce → Cₓ"]:::fusionStyle
        F2["Fusion Module 2<br/>Proj + Concat + Reduce → Cₓ"]:::fusionStyle
        F3["Fusion Module 3<br/>Proj + Concat + Reduce → Cₓ"]:::fusionStyle
        
        F0 -.-> F1 -.-> F2 -.-> F3
    end
    
    %% ============ DECODER ============
    subgraph DECODER["🔺 URBAN CONTEXT DECODER"]
        direction TB
        D3["Context Block 3<br/>(Deepest)"]:::decoderStyle
        D2["Context Block 2<br/>+ Upsample"]:::decoderStyle
        D1["Context Block 1<br/>+ Upsample"]:::decoderStyle
        D0["Context Block 0<br/>(Highest Resolution)"]:::decoderStyle
        
        D3 --> D2 --> D1 --> D0
    end
    
    %% ============ OUTPUT HEAD ============
    MSF["📊 Multi-Scale Fusion<br/>(Project + Aggregate)"]:::fusionStyle
    CLASSIFIER["1×1 Conv<br/>→ num_classes"]:::outputStyle
    UPSAMPLE["Bilinear Upsample<br/>→ Input Size"]:::processStyle
    OUTPUT["📈 Segmentation Map<br/>(B, num_classes, H, W)"]:::outputStyle
    
    %% ============ CONNECTIONS ============
    
    %% Input splits to both branches
    INPUT ==> SPATIAL
    INPUT ==> WAVELET
    
    %% Spatial to Fusion
    S0 --> F0
    S1 --> F1
    S2 --> F2
    S3 --> F3
    
    %% Wavelet to Fusion
    W0 --> F0
    W1 --> F1
    W2 --> F2
    W3 --> F3
    
    %% Fusion to Decoder
    F0 --> D0
    F1 --> D1
    F2 --> D2
    F3 --> D3
    
    %% Decoder to Multi-Scale Fusion
    D0 --> MSF
    D1 --> MSF
    D2 --> MSF
    D3 --> MSF
    
    %% Final Output Path
    MSF ==> CLASSIFIER ==> UPSAMPLE ==> OUTPUT
```

## Architecture Overview

### Key Components

1. **Dual Encoder Branches**
   - **Spatial Encoder**: Standard Vision Mamba (VSSM) backbone extracting spatial features at 4 scales
   - **Wavelet Encoder**: Haar decomposition + 4 independent VSSM encoders (one per subband) capturing frequency information

2. **Stage-wise Fusion Modules**
   - Fuse spatial (C) and wavelet (4×C) features at each scale
   - Operations: Project → Concatenate → Reduce → Mix → Output Cₓ channels

3. **Urban Context Decoder**
   - Hierarchical decoder with attention-based context blocks
   - Progressively upsamples and refines features from deep to shallow
   - Combines spatial + channel attention mechanisms

4. **Multi-Scale Fusion + Classifier**
   - Aggregates decoder outputs across all scales
   - 1×1 convolution produces per-pixel class logits
   - Bilinear upsampling to match input resolution

### Preview Instructions

- Open this file in VS Code and use the built-in Markdown preview
- For Mermaid rendering, install the **"Markdown Preview Mermaid Support"** extension
- Alternatively, view on GitHub or any Mermaid-compatible viewer
