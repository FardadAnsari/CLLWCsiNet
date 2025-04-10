# CLLWCsiNet Complete Architecture

```mermaid
%%{init: {'theme': 'neutral', 'fontFamily': 'Fira Code'}}%%
flowchart TD
    classDef module fill:#e1f5fe,stroke:#039be5,stroke-width:2px;
    classDef tensor fill:#e8f5e9,stroke:#43a047,stroke-width:2px;

    %% ========== MAIN INPUT/OUTPUT ==========
    IN([("Input<br/>(N,2,32,32)")]):::tensor
    OUT([("Output<br/>(N,2,32,32)")]):::tensor

    %% ========== ENCODER BLOCK ==========
    subgraph ENC["Encoder Block"]
        direction TB
        P1["[1,7]→[7,1]<br/>(N,2,32,32)"]:::module
        P2["[1,5]→[5,1]<br/>(N,2,32,32)"]:::module
        P3["[1,3]→[3,1]<br/>(N,2,32,32)"]:::module
        CCAT("Concat<br/>(N,6,32,32)"):::tensor
        C1x1["Conv1x1<br/>(N,2,32,32)"]:::module
    end

    %% ========== COMPRESSION BLOCK ==========
    subgraph COMP["Encoder_Compression"]
        direction LR
        IN_COMP(["(N,64,1,32)"]):::tensor
        P1_COMP["64→32→16→4<br/>(N,4,1,32)"]:::module
        P2_COMP["64→4<br/>(N,4,1,32)"]:::module
        CCAT_COMP("Concat<br/>(N,8,1,32)"):::tensor
        OUT_COMP["8→4<br/>(N,4,1,32)"]:::module
    end

    %% ========== DECODER BLOCK ==========
    subgraph DEC["Decoder Block"]
        direction LR
        IN_DEC(["(N,4,1,32)"]):::tensor
        UE["4→8→16→64<br/>(N,64,1,32)"]:::module
        AGN["remove_AGN<br/>4→8→16→64"]:::module
        SUB("Subtract<br/>(N,64,1,32)"):::tensor
    end

    %% ========== REFINER BLOCK ==========
    subgraph REF["RefineNet (x2)"]
        direction LR
        IN_REF(["(N,2,32,32)"]):::tensor
        CV7["Conv1x7 Branch<br/>(N,2,32,32)"]:::module
        CV5["Conv1x5 Branch<br/>(N,2,32,32)"]:::module
        CCAT_REF("Concat<br/>(N,4,32,32)"):::tensor
        CV1x1["Conv1x1<br/>(N,2,32,32)"]:::module
        RESIDUAL["+ Input"]:::module
    end

    %% ========== DATA FLOW ==========
    IN --> ENC
    ENC -->|"reshape"| IN_COMP
    IN_COMP --> COMP
    COMP -->|"(N,4,1,32)"| NOISE[["Add Noise<br/>SNR=40dB"]]
    NOISE -->|clone| IN_DEC
    IN_DEC --> DEC
    DEC -->|"reshape"| IN_REF
    IN_REF --> REF
    REF --> OUT

    %% ========== MODULE CONNECTIONS ==========
    ENC --> P1 & P2 & P3 --> CCAT --> C1x1
    COMP --> IN_COMP --> P1_COMP & P2_COMP --> CCAT_COMP --> OUT_COMP
    DEC --> UE & AGN --> SUB
    REF --> CV7 & CV5 --> CCAT_REF --> CV1x1 --> RESIDUAL