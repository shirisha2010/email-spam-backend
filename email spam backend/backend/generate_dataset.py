"""Generates a synthetic starter dataset (data/emails.csv) with columns: label,text.
Replace with a real dataset (e.g. Kaggle 'spam.csv' / Enron / SpamAssassin) for production use."""
import csv, random
random.seed(42)

spam_t = [
 "WINNER!! You have been selected to receive a ${amt} gift card. Click {link} to claim your prize now",
 "Congratulations! You won a free {prize}. Reply YES to claim. Limited time offer, act now!!!",
 "URGENT: Your account has been suspended. Verify your password immediately at {link}",
 "Get {pct}% OFF {prod} today only! Buy now, no prescription needed. Cheap prices guaranteed",
 "Make ${amt} per week working from home. No experience needed. Click here to start earning money fast",
 "Dear customer, your bank account is locked. Confirm your details at {link} to avoid closure",
 "You have inherited ${amt} from a distant relative. Send your bank details to receive the funds",
 "FREE entry in our weekly draw to win a {prize}! Text WIN to 80085 now. T&Cs apply",
 "Hot singles in your area want to meet you tonight. Click {link} for free membership",
 "Lowest mortgage rates! Refinance now and save thousands. Apply today at {link}",
 "Your PayPal payment of ${amt} is pending. Login at {link} to release the funds immediately",
 "Lose {pct} pounds in 7 days with this miracle pill! Order now, risk free, 100% guaranteed",
 "Claim your exclusive {prize} before midnight! Offer expires soon. Click {link} to redeem",
 "Dear friend, I am a prince and need your help transferring ${amt}. You will receive 30% commission",
 "Final notice: unpaid invoice of ${amt}. Pay immediately at {link} or face legal action",
 "Cheap {prod} online! Viagra, Cialis, discount pharmacy. No doctor visit. Order today",
 "Congratulations, you are our lucky visitor number 1,000,000! Collect your {prize} here {link}",
 "Earn bitcoin daily with our auto-trading bot. Invest ${amt} and get 10x returns guaranteed",
]
ham_t = [
 "Hi {name}, are we still meeting tomorrow at {time} to discuss the project plan?",
 "Please find attached the minutes of today's meeting. Let me know if I missed anything.",
 "Hey {name}, can you send me the report before Friday? Thanks a lot for your help.",
 "Reminder: the team standup is at {time} in the main conference room.",
 "Thanks for your email. I will review the document and get back to you by end of day.",
 "Dear Professor, I would like to request an extension for the assignment due next week.",
 "Your order #{num} has shipped and will arrive on Thursday. Track it in your account.",
 "Lunch at {time}? I was thinking of trying the new place near the office.",
 "Hi {name}, here are the notes from the lecture on machine learning. Hope they help with revision.",
 "Can we reschedule our call to {time}? Something came up with the client presentation.",
 "Happy birthday {name}! Hope you have a wonderful day. See you at the party this weekend.",
 "The quarterly budget review is scheduled for Monday. Please bring your department figures.",
 "I have pushed the latest code changes to the repository. Please review the pull request when free.",
 "Your interview has been scheduled for {time} on Tuesday. Please confirm your availability.",
 "Thanks for dinner last night, {name}. It was great catching up after so long.",
 "Attached is the invoice for last month's consulting services. Payment due within 30 days.",
 "The college library will be closed on Saturday for maintenance. Plan your study accordingly.",
 "Could you share the slides from yesterday's seminar? I would like to go through them again.",
]
fill = dict(
 amt=lambda: random.choice(["500","1,000","5,000","10,000","50,000","2,500,000"]),
 link=lambda: random.choice(["http://bit.ly/x{}".format(random.randint(100,999)),"www.claim-prize{}.com".format(random.randint(1,99)),"http://secure-login{}.net".format(random.randint(1,99))]),
 prize=lambda: random.choice(["iPhone 15","holiday trip","laptop","cash reward","luxury car","gift voucher"]),
 pct=lambda: random.choice(["50","70","80","90","20","30"]),
 prod=lambda: random.choice(["watches","medications","software","sunglasses","handbags"]),
 name=lambda: random.choice(["Rahul","Priya","Sam","Anita","John","Meera","David","Kiran"]),
 time=lambda: random.choice(["9 am","10:30","2 pm","4:15 pm","noon","5 pm"]),
 num=lambda: str(random.randint(10000,99999)),
)
def render(t):
    out = t
    for k,f in fill.items():
        while "{"+k+"}" in out: out = out.replace("{"+k+"}", f(), 1)
    return out
extra_ham = ["ok","sure see you then","thanks","I will call you later","Good morning team, please update the tracker"]
rows=[]
for _ in range(900): rows.append(("spam", render(random.choice(spam_t))))
for _ in range(1400): rows.append(("ham", render(random.choice(ham_t))))
for _ in range(100): rows.append(("ham", random.choice(extra_ham)))
random.shuffle(rows)
with open("data/emails.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["label","text"]); w.writerows(rows)
print("wrote", len(rows), "rows")
