#!/usr/bin/env python3
"""
简单邮件通知脚本：训练结束后调用，用于发送结果邮件。

你可以直接在本文件顶部修改邮箱配置，而不用在 shell 里 export 环境变量。

【请按需要修改下面这段 SMTP 配置】
"""

import argparse
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr


# ========= 在这里填写你的邮箱配置（只改这一段即可） =========
SMTP_HOST = "smtp.qq.com"  # 例如: smtp.qq.com / smtp.163.com
SMTP_PORT = 587                      # 常见为 587 或 465
SMTP_USER = "2432373417@qq.com"         # 例如: yourname@qq.com
SMTP_PASS = "olwzznyhvglvebad"       # 建议使用邮箱提供的“客户端授权码”
EMAIL_FROM = "2432373417@qq.com"           # 一般与 SMTP_USER 相同
EMAIL_TO = "919510240@qq.com"         # 可与 EMAIL_FROM 相同，也可以多个，用逗号分隔
# =====================================================


def send_mail(subject: str, body: str) -> None:
    host = SMTP_HOST
    port = int(SMTP_PORT)
    user = SMTP_USER
    password = SMTP_PASS
    email_from = EMAIL_FROM
    email_to = EMAIL_TO

    if not all([host, user, password, email_from, email_to]):
        raise RuntimeError("请先在 email_notify.py 顶部填写完整 SMTP_HOST/SMTP_USER/SMTP_PASS/EMAIL_FROM/EMAIL_TO 配置。")

    msg = MIMEText(body, _subtype="plain", _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr(("Brew-Train Bot", email_from))
    msg["To"] = email_to

    to_list = [addr.strip() for addr in email_to.split(",") if addr.strip()]

    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(email_from, to_list, msg.as_string())


def main():
    parser = argparse.ArgumentParser(description="发送简单训练结果邮件通知")
    parser.add_argument("--subject", required=True, help="邮件主题")
    parser.add_argument("--body", required=True, help="邮件正文")
    args = parser.parse_args()

    send_mail(args.subject, args.body)


if __name__ == "__main__":
    main()


