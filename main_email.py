import smtplib
import ssl

sender = 'lorenzobergadano@gmail.com'
receivers = ['lorenzobergadano@gmail.com']
message = """From: From Person <from@example.com>
To: To Person <to@example.com>
Subject: SMTP email example


This is a test message.
"""


def main():
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(sender, 'souy htkd fego mzmj')
            server.sendmail(sender, receivers, message)
            print("Successfully sent email")
    except smtplib.SMTPException:
        pass


if __name__ == '__main__':
    main()
