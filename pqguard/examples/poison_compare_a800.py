#!/usr/bin/env python3
"""
poison_compare_a800.py

对比场景：
- baseline: 未使用 PQC，直接加载“干净模型”和“被篡改模型”
- pqc: 使用 PQGuard（签名 + 验签 + 加密会话），对篡改模型进行阻断

- baseline_clean / baseline_tampered
- pqc_verify_clean / pqc_verify_tampered
- pqc_infer_plain_len / pqc_infer_enc_len
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM

from pqguard import PQGuardLLMPipeline
from pqguard.sal import PQSecurityAbstractionLayer
from pqguard.model_integrity import ModelIntegrityVerifier
from pqguard.session_encryption import SecureSessionManager


# 根据实际缓存路径调整
CLEAN_SNAPSHOT = Path(
    "/root/private_data/.cache/huggingface/hub/"
    "models--Qwen--Qwen1.5-7B-Chat/snapshots/5f4f5e69ac7f1d508f8369e977de208b4803444b"
)
TAMPERED_DIR = Path("/root/private_data/Yijunhao/models_poisoned/qwen1.5_tampered")
MODEL_ID = "Qwen/Qwen1.5-7B-Chat"

PROMPT = "简要说明什么是后量子密码学。"


def prepare_tampered_copy():
    """复制一份模型目录并做轻微篡改（用于模拟投毒/篡改）。"""
    if not CLEAN_SNAPSHOT.exists():
        raise FileNotFoundError(f"CLEAN_SNAPSHOT not found: {CLEAN_SNAPSHOT}")

    if not TAMPERED_DIR.exists():
        shutil.copytree(CLEAN_SNAPSHOT, TAMPERED_DIR)

    # 篡改 config.json（追加一行标记），保持 HF 仍能加载
    cfg = TAMPERED_DIR / "config.json"
    if cfg.exists():
        content = cfg.read_text(encoding="utf-8", errors="ignore")
        if "POISONED_MARKER" not in content:
            cfg.write_text(content + "\n// POISONED_MARKER", encoding="utf-8")


def run_baseline():
    """未使用 PQC 的 baseline：直接加载干净/篡改模型并推理。"""
    # clean
    try:
        tok_clean = AutoTokenizer.from_pretrained(str(CLEAN_SNAPSHOT), trust_remote_code=True)
        mdl_clean = AutoModelForCausalLM.from_pretrained(
            str(CLEAN_SNAPSHOT),
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        mdl_clean.eval()

        import torch  # 局部导入，避免顶部依赖

        inputs = tok_clean(PROMPT, return_tensors="pt").to(mdl_clean.device)
        with torch.no_grad():
            out = mdl_clean.generate(**inputs, max_new_tokens=64)
        txt_clean = tok_clean.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    except Exception as e:
        txt_clean = f"ERROR:{type(e).__name__}"

    # tampered
    try:
        tok_t = AutoTokenizer.from_pretrained(str(TAMPERED_DIR), trust_remote_code=True)
        mdl_t = AutoModelForCausalLM.from_pretrained(
            str(TAMPERED_DIR),
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        mdl_t.eval()

        import torch  # 局部导入

        inputs_t = tok_t(PROMPT, return_tensors="pt").to(mdl_t.device)
        with torch.no_grad():
            out_t = mdl_t.generate(**inputs_t, max_new_tokens=64)
        txt_t = tok_t.decode(out_t[0][inputs_t["input_ids"].shape[-1] :], skip_special_tokens=True)
    except Exception as e:
        txt_t = f"ERROR:{type(e).__name__}"

    # 只输出前 80 字符，避免过长
    print(f"baseline_clean: {txt_clean[:80]}")
    print(f"baseline_tampered: {txt_t[:80]}")


def run_pqc():
    """使用 PQGuard 进行签名 + 验签 + 加密推理，对篡改模型进行阻断。"""
    sal = PQSecurityAbstractionLayer()
    verifier = ModelIntegrityVerifier(sal)

    # 为干净模型创建 manifest + 签名密钥
    pub, sec = sal.generate_signature_keypair()
    verifier.add_trusted_key(MODEL_ID, pub)
    manifest = verifier.create_manifest(
        model_path=CLEAN_SNAPSHOT,
        model_id=MODEL_ID,
        model_type="qwen",
        version="1.0",
        secret_key=sec,
    )
    manifest_path = CLEAN_SNAPSHOT / "manifest.json"
    verifier.save_manifest(manifest, manifest_path)

    # 验证干净模型
    ok_clean, err_clean = verifier.verify_complete(CLEAN_SNAPSHOT, manifest_path)
    # 验证篡改模型（使用同一 manifest）
    ok_tampered, err_tampered = verifier.verify_complete(TAMPERED_DIR, manifest_path)

    print(f"pqc_verify_clean: {'OK' if ok_clean else 'FAIL'}")
    print(f"pqc_verify_tampered: {'OK' if ok_tampered else 'FAIL'}")

    # 仅对通过验签的干净模型进行加密推理
    pqc_plain_len = "NA"
    pqc_enc_len = "NA"
    if ok_clean:
        pipeline = PQGuardLLMPipeline(
            model_id=str(CLEAN_SNAPSHOT),
            model_type="qwen",
            verify_integrity=False,  # 已手动验签，这里不重复
            enable_encryption=True,
            enable_audit=False,
        )
        server_pk = pipeline.get_server_public_key()
        client_mgr = SecureSessionManager()
        kem_ct, _, client_key = client_mgr.establish_session(server_pk)
        session_id, _ = pipeline.establish_session(kem_ct)

        # 明文推理（仅用于统计长度，不走 PQC，加密对照）
        res_plain = pipeline.generate(
            prompt=PROMPT,
            session_id=None,
            max_new_tokens=64,
            temperature=0.7,
        )
        txt_plain = res_plain["response"]
        pqc_plain_len = len(txt_plain)

        # 加密会话推理
        res_enc = pipeline.generate(
            prompt=PROMPT,
            session_id=session_id,
            max_new_tokens=64,
            temperature=0.7,
        )
        if res_enc.get("encrypted"):
            ct_hex = res_enc["response"]
            pqc_enc_len = len(ct_hex)
        else:
            pqc_enc_len = 0

    print(f"pqc_infer_plain_len: {pqc_plain_len}")
    print(f"pqc_infer_enc_len: {pqc_enc_len}")


def main():
    if not CLEAN_SNAPSHOT.exists():
        print(f"err: clean_snapshot_not_found:{CLEAN_SNAPSHOT}")
        return

    prepare_tampered_copy()

    # baseline: 无 PQC
    run_baseline()

    # pqc: 有签名/验签 + 加密推理
    run_pqc()


if __name__ == "__main__":
    main()


