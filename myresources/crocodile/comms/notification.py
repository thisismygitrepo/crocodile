
"""Notifications Module
"""
# import numpy as np
# import matplotlib.pyplot as plt
import crocodile.toolbox as tb
import smtplib
import imaplib
# from email import message
# from email import encoders
# from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Any, Union

"""

"""


def get_gtihub_markdown_css(): return tb.P(r'https://raw.githubusercontent.com/sindresorhus/github-markdown-css/main/github-markdown-dark.css').download_to_memory().text


class Email:
    @staticmethod
    def get_source_of_truth(): return tb.P.home().joinpath("dotfiles/machineconfig/emails.ini").readit()

    def __init__(self, config: dict[str, Any]):
        self.config = config
        from smtplib import SMTP_SSL, SMTP
        self.server: Union[SMTP_SSL, SMTP]
        if config['encryption'].lower() == "ssl": self.server = smtplib.SMTP_SSL(host=self.config["smtp_host"], port=self.config["smtp_port"])
        elif config['encryption'].lower() == "tls": self.server = smtplib.SMTP(host=self.config["smtp_host"], port=self.config["smtp_port"])
        self.server.login(self.config['email_add'], password=self.config["password"])

    def send_message(self, to: str, subject: str, body: str, txt_to_html: bool = True, attachments: Optional[list[Any]] = None):
        _ = attachments
        body += "\n\nThis is an automated email sent via crocodile.comms script."
        # msg = message.EmailMessage()
        msg = MIMEMultipart("alternative")
        msg["subject"] = subject
        msg["From"] = self.config['email_add']
        msg["To"] = to
        # msg['Content-Type'] = "text/html"
        # msg.set_content(body)

        # <link rel="stylesheet" href="github-markdown.css">
        # <link type="text/css" rel="stylesheet" href="https://raw.githubusercontent.com/sindresorhus/github-markdown-css/main/github-markdown-dark.css" />

        if txt_to_html:  # add gtihub markdown stylesheet
            body = f"""
<!DOCTYPE html>
<html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{get_gtihub_markdown_css()}
    .markdown-body {{
        box-sizing: border-box;
        min-width: 200px;
        max-width: 980px;
        margin: 0 auto;
        padding: 45px;
        line-height: 1.8;
    }}
    @media (max-width: 767px) {{.markdown-body {{padding: 15px;}}
    }}
</style>
<body>
<div class="markdown-body">
{tb.install_n_import("markdown").markdown(body)}
</div>
</body>
</html>"""
        msg.attach(MIMEText(body, "html"))
        # if attachments is None: attachments = []  # see: https://fedingo.com/how-to-send-html-mail-with-attachment-using-python/
        # for attachment in attachmenthrs: msg.attach(attachment.read_bytes(), filename=attachment.stem, maintype="image", subtype=attachment.suffix)
        # for attachment in attachments: msg.attach(attachment.read_bytes(), filename=attachment.stem, maintype="application", subtype="octet-stream")

        self.server.send_message(msg)

    @staticmethod
    def manage_folders(email_add: str, pwd: str):
        server = imaplib.IMAP4()
        server.starttls()
        server.login(email_add, password=pwd)

    def send_email(self, to_addrs: str, msg: str): return self.server.sendmail(from_addr=self.config['email_add'], to_addrs=to_addrs, msg=msg)
    def close(self): self.server.quit()    # Closing is vital as many servers do not allow mutiple connections.

    @staticmethod
    def send_and_close(config_name: str, to: str, subject: str, msg: str):
        tmp = Email(config=Email.get_source_of_truth()[config_name]); tmp.send_message(to, subject, msg); tmp.close()


class PhoneNotification:  # security concerns: avoid using this.
    def __init__(self, bulletpoint_token: str):
        pushbullet = tb.install_n_import("pushbullet")
        self.api = pushbullet.Pushbullet(bulletpoint_token)
    def send_notification(self, title: str = "Note From Python", body: str = "A notfication"): self.api.push_note(title=title, body=body)
    @staticmethod
    def open_website(): tb.P(r"https://www.pushbullet.com/").readit()
    @staticmethod  # https://www.youtube.com/watch?v=tbzPcKRZlHg
    def try_me(bulletpoint_token: str): n = PhoneNotification(bulletpoint_token); n.send_notification()


if __name__ == '__main__':
    pass
