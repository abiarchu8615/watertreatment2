import requests
import smtplib
from email.mime.text import MIMEText

# =========================
# TELEGRAM ALERT FUNCTION
# =========================
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"


def send_telegram_alert(message: str):
    """
    Send alert to Telegram bot (Industrial IoT alert system)
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"🚨 WATER PLANT ALERT 🚨\n\n{message}"
    }

    try:
        requests.post(url, data=payload)
        print("Telegram alert sent")
    except Exception as e:
        print("Telegram error:", e)


# =========================
# EMAIL ALERT FUNCTION
# =========================
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "receiver_email@gmail.com"


def send_email_alert(subject: str, message: str):
    """
    Send email alert for industrial monitoring system
    """
    try:
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("Email alert sent")

    except Exception as e:
        print("Email error:", e)