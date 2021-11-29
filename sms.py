# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = 'AC147174888cada5d58041821c8e477d2c'
auth_token = '353ed3df37e886ef162fd4376040ba6c'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                     body="VAST-Net noticed something unusual, deploy the drone!",
                     from_='+18285547879',
                     to='+16262176595'
                 )

print(message.sid)


















