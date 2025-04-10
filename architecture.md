# CLLWCsiNet Architecture

```mermaid
flowchart TB
    %% ======================= MAIN NETWORK =======================
    subgraph CLLWCsiNet["CLLWCsiNet (Main Network)"]
        direction TB
        Input[(Input\n(N,2,32,32))] --> Encoder
        Encoder --> |(N,64,1,32)| Compressor
        Compressor --> |(N,4,1,32)| Noise[Add Noise\nSNR=40dB]
        Noise --> Decoder
        Decoder --> |(N,2,32,32)| Refiner
        Refiner --> Output[(Output\n(N,2,32,32))]
    end

    %% ======================= MODULES =======================
    subgraph Encoder["Encoder Block"]
        direction TB
        P1["P1: [1,7]→[7,1]\n(N,2,32,32)"]
        P2["P2: [1,5]→[5,1]\n(N,2,32,32)"]
        P3["P3: [1,3]→[3,1]\n(N,2,32,32)"]
        Concat["Concat\n(N,6,32,32)"]
        Conv1x1["Conv1x1\n(N,2,32,32)"]
        
        Input -.-> P1 & P2 & P3
        P1 & P2 & P3 --> Concat --> Conv1x1
    end

    subgraph Compressor["Encoder_Compression"]
        direction LR
        InComp[(Input\n(N,64,1,32))]
        Path1["64→32→16→4\n(N,4,1,32)"]
        Path2["64→4\n(N,4,1,32)"]
        Merge["Concat\n(N,8,1,32)"]
        Final["8→4\n(N,4,1,32)"]
        
        InComp --> Path1 & Path2
        Path1 & Path2 --> Merge --> Final
    end

    subgraph Decoder["Decoder Block"]
        direction LR
        InDec[(Input\n(N,4,1,32))]
        UE["UE Path: 4→8→16→64\n(N,64,1,32)"]
        AGN["remove_AGN: 4→8→16→64\n(N,64,1,32)"]
        Subtract["Subtract\n(N,64,1,32)"]
        
        InDec --> UE & AGN
        UE & AGN --> Subtract
    end

    subgraph Refiner["RefineNet (x2)"]
        direction LR
        InRef[(Input\n(N,2,32,32))]
        Conv7["Conv1x7 Branch\n(N,2,32,32)"]
        Conv5["Conv1x5 Branch\n(N,2,32,32)"]
        ConcatRef["Concat\n(N,4,32,32)"]
        Conv1x1Ref["Conv1x1\n(N,2,32,32)"]
        Residual["+ Input\n(N,2,32,32)"]
        
        InRef --> Conv7 & Conv5 --> ConcatRef --> Conv1x1Ref --> Residual
    end

    %% ======================= CONNECTIONS =======================
    Encoder -->|reshape| Compressor
    Noise -->|clone| Decoder
    Decoder -->|reshape| Refiner