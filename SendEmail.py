import smtplib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--acc', default=0.0, type=float)
parser.add_argument('--mll', default=0.0, type=float)

flags = parser.parse_args()

server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
server.login("mmerghaney@aimsammi.org", "M@r@AMMI01")
msg = '''\
Subject: ResConvGP model training
        
Congratulation your training model has been finished with accuracy {:.5f} and mll {:.7f}.

Best Regards,
        '''.format(flags.acc, flags.mll)
server.sendmail(
  "mmerghaney", 
  "mmerghaney@aimsammi.org", 
  msg)
server.quit()