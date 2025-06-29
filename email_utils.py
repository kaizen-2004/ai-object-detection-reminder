import smtplib
from email.message import EmailMessage

def send_email(subject, body, to_email, from_email, app_password):
    """
    Sends an email using Gmail's SMTP server.

    Returns:
        True if sent successfully, False otherwise.
    """
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        msg.set_content(body)

        # Connect to Gmail's SMTP server
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(from_email, app_password)
            smtp.send_message(msg)

        return True

    except Exception as e:
        print(f"Email sending failed: {e}")
        return False
