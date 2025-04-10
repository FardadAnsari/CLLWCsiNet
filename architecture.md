```mermaid
flowchart TB
    subgraph CLLWCsiNet["CLLWCsiNet (Main Network)"]
        direction TB
        Input[(2,32,32)] --> Encoder
        Encoder --> |(64,1,32)| Compressor
        Compressor --> |(4,1,32)| Noise
        Noise --> Decoder
        Decoder --> |(2,32,32)| Refiner
        Refiner --> Output[(2,32,32)]
    end

    subgraph Encoder["Encoder Block"]
        P1["P1: [1,7]→[7,1]"]
        P2["P2: [1,5]→[5,1]"]
        P3["P3: [1,3]→[3,1]"]
        Concat["Concat (6,32,32)"]
        Conv1x1["Conv1x1 (2,32,32)"]
    end

    subgraph Compressor["Encoder_Compression"]
        Path1["64→32→16→4"]
        Path2["64→4"]
        Merge["Concat (8,1,32)"]
        Final["8→4"]
    end

    subgraph Decoder["Decoder Block"]
        UE["UE Path: 4→8→16→64"]
        AGN["remove_AGN: 4→8→16→64"]
        Subtract["Subtract"]
    end

    subgraph Refiner["RefineNet x2"]
        Conv1["Conv1x7 (2→8)"]
        Conv2["Conv1x5 (2→8)"]
        Merge["Concat (4,32,32)"]
        Residual["Residual Add"]
    end