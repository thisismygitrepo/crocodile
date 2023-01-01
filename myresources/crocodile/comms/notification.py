
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


"""

"""


def get_gtihub_markdown_css(): return tb.P(r'https://raw.githubusercontent.com/sindresorhus/github-markdown-css/main/github-markdown-dark.css').download(memory=True).text


class Email:
    @staticmethod
    def get_source_of_truth(): return tb.P.home().joinpath("dotfiles/creds/msc/source_of_truth.py").readit(strict=False)

    def __init__(self, config):
        self.config = config
        self.server = smtplib.SMTP(host=self.config["smpt_host"], port=self.config["smpt_port"])
        self.server.login(self.config['email_add'], password=self.config["get_password"]())

    def send_message(self, to, subject, body, txt_to_html=True, attachments=None):
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
    def manage_folders(email_add, pwd):
        server = imaplib.IMAP4()
        server.starttls()
        server.login(email_add, password=pwd)

    def send_email(self, to_addrs, msg): return self.server.sendmail(from_addr=self.config['email_add'], to_addrs=to_addrs, msg=msg)
    def close(self): self.server.quit()    # Closing is vital as many servers do not allow mutiple connections.

    @staticmethod
    def send_and_close(config_name, to, subject, msg): tmp = Email(config=Email.get_source_of_truth().EMAIL[config_name]); tmp.send_message(to, subject, msg); tmp.close()


class PhoneNotification:  # security concerns: avoid using this.
    def __init__(self, bulletpoint_token):
        pushbullet = tb.install_n_import("pushbullet")
        self.api = pushbullet.Pushbullet(bulletpoint_token)
    def send_notification(self, title="Note From Python", body="A notfication"): self.api.push_note(title=title, body=body)
    @staticmethod
    def open_website(): tb.P(r"https://www.pushbullet.com/").readit()
    @staticmethod  # https://www.youtube.com/watch?v=tbzPcKRZlHg
    def try_me(bulletpoint_token): n = PhoneNotification(bulletpoint_token); n.send_notification()


if __name__ == '__main__':
    pass
