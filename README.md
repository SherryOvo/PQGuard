# PQGuard: Post-Quantum Secure LLM Inference Framework

PQGuard 是第一个端到端的后量子安全架构，将 NIST 标准化的后量子密码学（PQC）方案集成到完整的 LLM 生命周期中。

## 功能特性

### 核心安全功能

1. **模型完整性验证**
   - 使用 CRYSTALS-Dilithium 对模型权重、LoRA 适配器和提示模板进行签名
   - SHA-256 哈希验证确保文件完整性
   - 支持细粒度的模型组件验证

2. **加密推理会话**
   - CRYSTALS-Kyber-768 密钥封装机制（KEM）进行密钥交换
   - AES-256-GCM 对称加密保护推理数据
   - 低延迟的端到端加密通信

3. **不可否认性审计日志**
   - SPHINCS+-SHA256 签名的审计日志
   - 支持时间范围、会话、模型等查询
   - 可锚定到不可变存储（WORM/区块链）

4. **混合模式支持**
   - 支持经典密码学与后量子密码学的混合使用
   - 平滑过渡期间的兼容性

## 架构组件

```
PQGuard/
├── sal.py                    # PQ-SAL: 后量子安全抽象层
├── model_integrity.py        # 模型完整性验证
├── session_encryption.py     # 会话加密管理
├── audit_log.py              # 审计日志系统
├── llm_pipeline.py           # LLM 推理管道集成
└── examples/                 # 示例代码
    ├── basic_usage.py
    ├── create_model_manifest.py
    ├── audit_log_demo.py
    └── complete_demo.py
```

## 安装要求

### 系统依赖

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake libssl-dev python3-dev

# Python 包
pip install pyoqs cryptography transformers accelerate torch
```

### 安装 PyOQS（如果未安装）

```bash
git clone https://github.com/open-quantum-safe/liboqs-python.git
cd liboqs-python
pip install .
```

## 快速开始

### 1. 基本使用

```python
from pqguard import PQGuardLLMPipeline

# 初始化管道
pipeline = PQGuardLLMPipeline(
    model_id="/path/to/model",
    model_type="qwen",
    enable_encryption=True,
    enable_audit=True,
)

# 建立加密会话
server_pk = pipeline.get_server_public_key()
# ... 客户端使用 KEM 封装密钥 ...
session_id, _ = pipeline.establish_session(kem_ciphertext)

# 安全推理
result = pipeline.generate(
    prompt="你的问题",
    session_id=session_id,
    max_new_tokens=512,
)
```

### 2. 创建模型清单

```python
from pqguard.model_integrity import ModelIntegrityVerifier
from pqguard.sal import PQSecurityAbstractionLayer

sal = PQSecurityAbstractionLayer()
verifier = ModelIntegrityVerifier(sal)

# 生成签名密钥对
public_key, secret_key = sal.generate_signature_keypair()

# 创建并签名清单
manifest = verifier.create_manifest(
    model_path=Path("/path/to/model"),
    model_id="Qwen/Qwen1.5-7B-Chat",
    model_type="qwen",
    version="1.0",
    secret_key=secret_key,
)

# 保存清单
verifier.save_manifest(manifest, Path("manifest.json"))
```

### 3. 运行示例

```bash
# 基本使用示例
python pqguard/examples/basic_usage.py

# 创建模型清单
python pqguard/examples/create_model_manifest.py

# 审计日志演示
python pqguard/examples/audit_log_demo.py

# 完整演示
python pqguard/examples/complete_demo.py
```

## 安全假设

### 威胁模型

PQGuard 在量子增强的 Dolev-Yao 威胁模型下工作，假设：
- 攻击者具有量子计算能力
- 网络通信可能被窃听和篡改
- 模型文件可能被替换或修改
- 需要审计和不可否认性保证

### 安全属性

1. **机密性 (IND-CCA2)**
   - 基于 Module-LWE 假设（Kyber）
   - 抵抗量子攻击的密钥交换
   - AES-256-GCM 提供认证加密

2. **完整性 (EUF-CMA)**
   - 基于 Module-SIS 假设（Dilithium）
   - 模型文件的签名验证
   - 抵抗量子攻击的数字签名

3. **不可否认性**
   - SPHINCS+ 签名提供不可否认性
   - 审计日志的法律约束力
   - 不可变存储锚定

## API 文档

### PQSecurityAbstractionLayer

后量子安全抽象层，提供统一的 PQC 操作接口。

#### 主要方法

- `generate_kem_keypair()` - 生成 Kyber 密钥对
- `kem_encapsulate(public_key)` - 密钥封装（客户端）
- `kem_decapsulate(secret_key, ciphertext)` - 密钥解封装（服务器）
- `generate_signature_keypair()` - 生成 Dilithium 签名密钥对
- `sign(data, secret_key)` - 签名数据
- `verify(data, signature, public_key)` - 验证签名

### ModelIntegrityVerifier

模型完整性验证器。

#### 主要方法

- `create_manifest(model_path, ...)` - 创建模型清单
- `verify_manifest(manifest)` - 验证清单签名
- `verify_model_files(model_path, manifest)` - 验证文件哈希

### SecureSessionManager

安全会话管理器。

#### 主要方法

- `get_server_public_key()` - 获取服务器公钥
- `create_session(kem_ciphertext)` - 创建会话（服务器）
- `establish_session(server_pk)` - 建立会话（客户端）
- `encrypt_request(session_id, plaintext)` - 加密请求
- `decrypt_response(session_key, ciphertext)` - 解密响应

### AuditLogger

审计日志记录器。

#### 主要方法

- `create_entry(...)` - 创建审计条目
- `log(entry)` - 记录条目
- `query_logs(...)` - 查询日志
- `verify_log_file(log_file)` - 验证日志文件

### PQGuardLLMPipeline

完整的 PQGuard LLM 推理管道。

#### 主要方法

- `generate(prompt, session_id, ...)` - 生成响应
- `get_server_public_key()` - 获取服务器公钥
- `establish_session(kem_ciphertext)` - 建立会话

## 性能考虑

- **密钥交换**: Kyber-768 密钥交换约 1-2ms
- **签名验证**: Dilithium-2 验证约 1-5ms
- **加密开销**: AES-GCM 加密/解密约 0.1-1ms/100KB
- **审计日志**: SPHINCS+ 签名约 10-50ms（批量签名可优化）

## 平台支持

-  Intel Xeon 服务器
-  NVIDIA Jetson AGX Orin
-  Apple M 系列（ARM64）
-  支持 CUDA 的 GPU 加速推理

## 测试模型

已测试的模型：
- Qwen1.5-7B-Chat
- Qwen2-VL-7B-Instruct
- Llama-3 (理论支持)
- Phi-3 (理论支持)

## 许可证

本项目基于开源许可证，具体见 LICENSE 文件。

## 引用

如果使用 PQGuard，请引用：

```bibtex
@article{pqguard2024,
  title={PQGuard: Post-Quantum Secure LLM Inference Framework},
  author={...},
  journal={...},
  year={2024}
}
```

## 贡献

欢迎贡献！请提交 Issue 或 Pull Request。

## 联系方式

如有问题或建议，请通过 Issue 联系。


