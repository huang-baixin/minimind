# MiniMind 项目工作流程

## 核心训练流程

```mermaid
flowchart TD
    A[📊 数据准备] --> B[🔤 分词器训练<br/>scripts/train_tokenizer.py]
    B --> C[🏋️ 预训练<br/>trainer/train_pretrain.py]
    C --> D[🎯 监督微调<br/>trainer/train_full_sft.py]
    
    D --> E{选择后续训练方式}
    E --> F[⚡ LoRA微调<br/>trainer/train_lora.py]
    E --> G[❤️ DPO训练<br/>trainer/train_dpo.py]
    E --> H[🧠 模型蒸馏<br/>trainer/train_distillation.py]
    
    G --> I[🤖 PPO强化学习<br/>trainer/train_ppo.py]
    I --> J[👥 GRPO训练<br/>trainer/train_grpo.py]
    J --> K[🔍 SPO训练<br/>trainer/train_spo.py]
    
    H --> L[💭 推理模型蒸馏<br/>trainer/train_distill_reason.py]
    
    F --> M[📈 模型评估<br/>eval_llm.py]
    K --> M
    L --> M
    
    M --> N[🚀 模型部署]
    
    subgraph "部署选项"
        N --> O1[🌐 Web演示<br/>scripts/web_demo.py]
        N --> O2[🔌 API服务<br/>scripts/serve_openai_api.py]
        N --> O3[🔄 模型转换<br/>scripts/convert_model.py]
        N --> O4[💬 聊天接口<br/>scripts/chat_openai_api.py]
    end
```

## 项目模块依赖关系

```mermaid
graph TB
    subgraph "核心模块"
        MODEL[model/model_minimind.py<br/>核心模型架构]
        CONFIG[MiniMindConfig<br/>模型配置]
        DATASET[dataset/lm_dataset.py<br/>数据集处理]
    end
    
    subgraph "训练流程"
        TRAINER_UTILS[trainer/trainer_utils.py<br/>训练工具]
        PRETRAIN[预训练]
        SFT[监督微调]
        RL[强化学习系列]
        DISTILL[模型蒸馏]
    end
    
    subgraph "工具脚本"
        TOKENIZER_TRAIN[分词器训练]
        MODEL_CONVERT[模型转换]
        API_SERVE[API服务]
        WEB_DEMO[Web界面]
    end
    
    %% 依赖关系
    MODEL --> PRETRAIN
    CONFIG --> MODEL
    DATASET --> PRETRAIN
    TRAINER_UTILS --> PRETRAIN
    PRETRAIN --> SFT
    SFT --> RL
    SFT --> DISTILL
    
    TOKENIZER_TRAIN --> PRETRAIN
    MODEL --> MODEL_CONVERT
    MODEL_CONVERT --> API_SERVE
    API_SERVE --> WEB_DEMO
    
    classDef core fill:#e3f2fd,stroke:#1976d2
    classDef train fill:#f3e5f5,stroke:#7b1fa2
    classDef tool fill:#e8f5e8,stroke:#388e3c
    
    class MODEL,CONFIG,DATASET core
    class PRETRAIN,SFT,RL,DISTILL,TRAINER_UTILS train
    class TOKENIZER_TRAIN,MODEL_CONVERT,API_SERVE,WEB_DEMO tool
```

## 技术栈架构

```mermaid
graph LR
    subgraph "深度学习框架"
        PYTORCH[PyTorch 2.6.0]
        TRANSFORMERS[Transformers 4.57.1]
    end
    
    subgraph "训练优化"
        TRL[TRL 0.13.0]
        PEFT[PEFT 0.7.1]
        WANDB[WandB/SwanLab]
    end
    
    subgraph "数据处理"
        DATASETS[HuggingFace Datasets]
        TOKENIZER[自定义分词器]
    end
    
    subgraph "部署工具"
        STREAMLIT[Streamlit]
        FLASK[Flask API]
        OPENAI[OpenAI兼容接口]
    end
    
    PYTORCH --> TRANSFORMERS
    TRANSFORMERS --> TRL
    TRANSFORMERS --> PEFT
    DATASETS --> TOKENIZER
    
    TRL --> WANDB
    PEFT --> WANDB
    
    STREAMLIT --> FLASK
    FLASK --> OPENAI
    
    classDef framework fill:#fff3e0,stroke:#f57c00
    classDef training fill:#e8f5e8,stroke:#43a047
    classDef data fill:#f3e5f5,stroke:#8e24aa
    classDef deploy fill:#e1f5fe,stroke:#0288d1
    
    class PYTORCH,TRANSFORMERS framework
    class TRL,PEFT,WANDB training
    class DATASETS,TOKENIZER data
    class STREAMLIT,FLASK,OPENAI deploy
```

## 文件组织结构

```
minimind/
├── 📁 model/                 # 模型架构
│   ├── model_minimind.py     # 核心模型
│   ├── model_lora.py         # LoRA实现
│   └── tokenizer配置         # 分词器
├── 📁 trainer/               # 训练流程
│   ├── train_pretrain.py     # 预训练
│   ├── train_full_sft.py     # 监督微调
│   ├── train_lora.py         # LoRA微调
│   ├── train_dpo.py          # DPO训练
│   ├── train_ppo.py          # PPO强化学习
│   ├── train_grpo.py         # GRPO训练
│   ├── train_spo.py          # SPO训练
│   ├── train_distillation.py # 模型蒸馏
│   └── train_distill_reason.py # 推理蒸馏
├── 📁 scripts/               # 工具脚本
│   ├── train_tokenizer.py    # 分词器训练
│   ├── convert_model.py      # 模型转换
│   ├── serve_openai_api.py   # API服务
│   ├── chat_openai_api.py    # 聊天接口
│   └── web_demo.py           # Web演示
├── 📁 dataset/               # 数据集
│   └── lm_dataset.py         # 数据集处理
├── eval_llm.py               # 模型评估
└── requirements.txt          # 依赖管理
```

这个工作流程展示了MiniMind项目从数据准备到模型部署的完整生命周期，突出了其模块化设计和完整的训练流程支持。