# CLLWCsiNet Architecture
```mermaid
flowchart TD
    classDef input fill:#4CAF50,stroke:#388E3C,color:white
    classDef output fill:#8BC34A,stroke:#689F38
    classDef module fill:#2196F3,stroke:#1565C0,color:white

    IN([["Input<br/><b>(N,2,32,32)</b>"]]):::input
    OUT([["Output<br/><b>(N,2,32,32)</b>"]]):::output

    subgraph ENC["Encoder Block"]
        P1["[1,7]→[7,1]"]:::module
        P2["[1,5]→[5,1]"]:::module
    end

    IN --> ENC --> OUT