# CLLWCsiNet Architecture

<pre>
Input (N,2,32,32)
       │
       ├───────────────┐
       │               │
┌──────▼──────┐   ┌────▼──────┐   ┌───────┐
│Encoder_P1   │   │Encoder_P2 │   │Encoder│
│[1,7]→[7,1] │   │[1,5]→[5,1]│   │_P3    │
└──────┬──────┘   └────┬──────┘   │[1,3]→│
       │               │          │[3,1] │
       └───────┬───────┘          └──┬───┘
               │                     │
           ┌───▼───┐                 │
           │Concat │                 │
           │(dim=1)◄────────────────┘
           └───┬───┘
               │
           ┌───▼───┐
           │Conv1x1│
           │(6→2)  │
           └───┬───┘
               │
           ┌───▼───┐
           │Reshape│
           │to     │
           │(N,64, │
           │1,32)  │
           └───┬───┘
               │
       ┌───────▼───────┐
       │Encoder_       │
       │Compression    │
       │(Parallel     │
       │ 64→32→16→4   │
       │ and 64→4)    │
       └───────┬───────┘
               │
       ┌───────▼───────┐
       │Add Noise      │
       │(SNR=40dB)     │
       └───────┬───────┘
               │
       ┌───────▼───────┐   ┌─────────────┐
       │Decoder_UE     │   │remove_AGN   │
       │(4→8→16→64)    │   │(4→8→16→64)  │
       └───────┬───────┘   └──────┬──────┘
               │                 │
               └───────┬─────────┘
                       │
                   ┌───▼───┐
                   │x - y  │
                   │(Residual)
                   └───┬───┘
                       │
                   ┌───▼───┐
                   │Reshape│
                   │to     │
                   │(N,2,32│
                   │,32)   │
                   └───┬───┘
                       │
               ┌───────▼───────┐
               │RefineNet      │
               │Blocks (x2)    │
               │(N,2,32,32)    │
               └───────┬───────┘
                       │
                   ┌───▼───┐
                   │Sigmoid│
                   │Output │
                   └───────┘
</pre>

## Key
- **Input/Output**: (N,2,32,32)
- **Concat**: Channel-wise concatenation
- **Reshape**: Changes tensor layout without data loss
- **Residual**: Skip connection adds input to output
