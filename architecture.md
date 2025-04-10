```mermaid
graph LR
    A[Input (N,2,32,32)] --> B[Encoder_P1]
    A --> C[Encoder_P2]
    A --> D[Encoder_P3]
    B --> E[Concat]
    C --> E
    D --> E
    E --> F[Conv1x1]
    F --> G[Reshape]
    G --> H[Encoder_Compression]
    H --> I[Add Noise]
    I --> J[Decoder_UE]
    I --> K[remove_AGN]
    J --> L[Subtract]
    K --> L
    L --> M[Reshape]
    M --> N[RefineNet x2]
    N --> O[Sigmoid]

    subgraph Encoder_P1
        B1[1x7 Conv] --> B2[7x1 Conv]
    end
    subgraph Encoder_P2
        C1[1x5 Conv] --> C2[5x1 Conv]
    end
    subgraph Encoder_Compression
        H1[64→32→16→4] --> H2[Concat]
        H3[64→4] --> H2
        H2 --> H4[8→4]
    end