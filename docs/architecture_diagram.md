
# CLLWCsiNet Architecture

```mermaid
flowchart TD
    classDef input fill:#4CAF50,stroke:#388E3C,color:white
    classDef output fill:#8BC34A,stroke:#689F38
    classDef module fill:#2196F3,stroke:#1565C0,color:white

    IN(("Input (N,2,32,32)")):::input
    OUT(("Output (N,2,32,32)")):::output

    subgraph ENC["Encoder Block"]
        P1["[1,7]→[7,1]"]:::module
        P2["[1,5]→[5,1]"]:::module
        P3["[1,3]→[3,1]"]:::module
    end

    subgraph COMP["Encoder_Compression"]
        C1["64→32→16→4"]:::module
        C2["64→4"]:::module
    end

    IN --> ENC --> COMP --> OUT