import smtplib, ssl

def send_email(receiver_email, title, body):
    port = 465  # For SSL
    smtp_host="smtp.gmail.com"
    password = "xxx"
    sender_email="xxx@gmail.com"

    # Create a secure SSL context
    context = ssl.create_default_context()
    message=f"""Subject: {title}

    {body}
    """

    with smtplib.SMTP_SSL(smtp_host, port, context=context) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

# commands used in solution video for reference
if __name__ == '__main__':
    # replace receiver email address
    send_email('RECEIVER@EMAIL.COM', 'Notification', 'Everything is awesome!')
