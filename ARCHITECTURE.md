# CLLWCsiNet Complete Architecture

<pre class="mermaid">
%%{init: {'theme': 'default', 'fontFamily': 'Arial', 'gantt': {'barHeight': 20}}}%%
flowchart TD
    classDef input fill:#4CAF50,stroke:#388E3C,color:white,stroke-width:2px;
    classDef output fill:#8BC34A,stroke:#689F38,stroke-width:2px;
    classDef module fill:#2196F3,stroke:#1565C0,color:white,stroke-width:2px;
    classDef tensor fill:#FFC107,stroke:#FFA000,stroke-width:2px;
    classDef noise fill:#9E9E9E,stroke:#616161,color:white,stroke-width:2px;

    %% ===== MAIN FLOW =====
    IN([["Input<br/><b>(N,2,32,32)</b>"]]):::input
    OUT([["Output<br/><b>(N,2,32,32)</b>"]]):::output

    %% ===== ENCODER =====
    subgraph ENC["<b>Encoder Block</b>"]
        direction TB
        P1["<b>[1,7]→[7,1]</b><br/>(N,2,32,32)"]:::module
        P2["<b>[1,5]→[5,1]</b><br/>(N,2,32,32)"]:::module
        P3["<b>[1,3]→[3,1]</b><br/>(N,2,32,32)"]:::module
        CCAT(["<b>Concat</b><br/>(N,6,32,32)"]):::tensor
        C1x1["<b>Conv1x1</b><br/>(N,2,32,32)"]:::module
    end

    %% ===== COMPRESSION =====
    subgraph COMP["<b>Encoder_Compression</b>"]
        direction LR
        IN_COMP([["<b>(N,64,1,32)</b>"]]):::tensor
        P1_COMP["<b>64→32→16→4</b><br/>(N,4,1,32)"]:::module
        P2_COMP["<b>64→4</b><br/>(N,4,1,32)"]:::module
        CCAT_COMP([["<b>Concat</b><br/>(N,8,1,32)"]]):::tensor
        OUT_COMP["<b>8→4</b><br/>(N,4,1,32)"]:::module
    end

    %% ===== NOISE INJECTION =====
    NOISE[["<b>Add Noise</b><br/>SNR=40dB"]]:::noise

    %% ===== DECODER =====
    subgraph DEC["<b>Decoder Block</b>"]
        direction LR
        IN_DEC([["<b>(N,4,1,32)</b>"]]):::tensor
        UE["<b>4→8→16→64</b><br/>(N,64,1,32)"]:::module
        AGN["<b>remove_AGN</b><br/>4→8→16→64"]:::module
        SUB([["<b>Subtract</b><br/>(N,64,1,32)"]]):::tensor
    end

    %% ===== REFINER =====
    subgraph REF["<b>RefineNet (x2)</b>"]
        direction LR
        IN_REF([["<b>(N,2,32,32)</b>"]]):::tensor
        CV7["<b>Conv1x7 Branch</b><br/>(N,2,32,32)"]:::module
        CV5["<b>Conv1x5 Branch</b><br/>(N,2,32,32)"]:::module
        CCAT_REF([["<b>Concat</b><br/>(N,4,32,32)"]]):::tensor
        CV1x1["<b>Conv1x1</b><br/>(N,2,32,32)"]:::module
        RESIDUAL["<b>+ Input</b>"]:::module
    end

    %% ===== CONNECTIONS =====
    IN --> ENC -->|"<b>reshape</b>"| IN_COMP --> COMP --> NOISE
    NOISE -->|"<b>clone</b>"| IN_DEC --> DEC -->|"<b>reshape</b>"| IN_REF --> REF --> OUT

    ENC --> P1 & P2 & P3 --> CCAT --> C1x1
    COMP --> P1_COMP & P2_COMP --> CCAT_COMP --> OUT_COMP
    DEC --> UE & AGN --> SUB
    REF --> CV7 & CV5 --> CCAT_REF --> CV1x1 --> RESIDUAL
</pre>

<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true, theme:'default'});</script>
