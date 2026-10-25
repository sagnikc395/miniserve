- llm serving requires to build software to serve the LLM to most users and techniques like paged attention and continuous batching help us regarding that.

```mermaid
graph LR
    subgraph Inputs ["Inputs Cluster"]
        ReqA["Request A"]
        ReqB["Request B"]
        ReqC["Request C"]
    end

    ReqA --> ClientNode["Client"]
    ReqB --> ClientNode
    ReqC --> ClientNode
    
    ClientNode --> Server["Server<br/>FastAPI"]
    Server --> Scheduler["Scheduler"]
    Scheduler --> Engine["Engine"]
    Engine --> Model["Model Forward Pass"]
    Model --> ClientNode
```

- Scheduler:
    - gets lots of requests 
    - where do these requests go 

- Engine:
    - how does it move the things forward 

- Model Forward Pass:
    - how to compute the things and maintain KV Cache for this 

- MLP is the easy part, at each time step, gets the same exact matrix multiplication 
$Q(GELU(W(x)))$

- attention in one picture

```mermaid
graph LR
    %% Styles
    classDef yellow fill:#f59e0b,stroke:#000,stroke-width:1px,color:#000;
    classDef red fill:#ef4444,stroke:#000,stroke-width:1px,color:#fff;
    classDef blue fill:#06b6d4,stroke:#000,stroke-width:1px,color:#fff;
    classDef purple fill:#a855f7,stroke:#000,stroke-width:1px,color:#fff;

    %% Nodes
    Embeddings["EMBEDDINGS"]:::yellow
    Q["Q<br/>(Queries)"]:::red
    KT["Kᵀ<br/>(Keys Transposed)"]:::blue
    AttnMatrix["ATTENTION MATRIX<br/>(A)"]:::purple
    V["VALUES<br/>(V)"]:::purple
    Outputs["OUTPUTS"]:::yellow

    %% Connections / Flow
    Embeddings -- "nn.Linear(768, 768)" --> Q
    Embeddings -- "nn.Linear(768, 768)" --> KT
    Embeddings -- "nn.Linear(768, 768)" --> V

    Q & KT -->|"Dot Product / Scale"| AttnMatrix
    AttnMatrix & V -->|"Matrix Multiplication"| Outputs

```